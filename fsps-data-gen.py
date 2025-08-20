import os

path = "./fsps"
os.environ["SPS_HOME"] = path

import jax
from jax import random
from jax import numpy as jnp
import fsps
import tqdm
from mpi4py import MPI as mpi

rank = mpi.COMM_WORLD.Get_rank()
size = mpi.COMM_WORLD.Get_size()
comm = mpi.COMM_WORLD

key = random.PRNGKey(rank)

dataset = 3

def fsps_call(params):
    # zcontinuous=1 is about how metalicity is handled
    # i think 1 or 2 is recommended
    sp = fsps.StellarPopulation(zcontinuous=1)
    sp.params["imf_type"] = params['imf_type']  # Chabrier
    sp.params["logzsol"] = params['logzsol']

    if dataset == 2 or dataset == 3:
        sp.params["dust_type"] = params['dust_type']
    
    if dataset == 3:
        sp.params['fburst'] = params['fburst']
        sp.params['tburst'] = params['tburst']
        sp.params['gas_logu'] = params['gas_logu']
        sp.params['imf_type'] = params['imf_type']  # 0: Salpeter, 1: Chabrier, 2: Kroupa
        sp.params['gas_logz'] = params['gas_logz']  # gas metallicity is the same as stellar metallicity
        sp.params["dust1"] = params['dust1']  # the Av dust parameter... dust one is general extinction from dust around star forming regions



    sp.params["dust2"] = params['dust2']  # the Av dust parameter... dust two is general extinction from dust around the galaxy
    # dust1 is dust around star forming regions
    sp.params["sfh"] = params["sfh"] 
    sp.params["add_neb_emission"] = params['add_neb_emission']  # whether to add nebular emission
    sp.params["add_neb_continuum"] = params['add_neb_continuum']  # whether to add nebular continuum
    if params['sfh'] == 4:  # if the sfh is tau model
        sp.params["tau"] = params['tau']
    
    #if params['sfh'] == 1:  # if the sfh is constant
    if 'const' in params:
        sp.params['const'] = params['const']

    # luminosity is solar luminosities per angstrom and wavelength is in Angstroms
    # spectra is galaxies rest frame
    lam, spec = sp.get_spectrum(tage=params['tage'], peraa=True)

    return lam, spec

#@jax.jit
def prior(key):

    subkeys = iter(jax.random.split(key, 20))  # increase if more draws needed

    parameters = {}
    if dataset == 1:
        parameters['dust2'] = 0.0
        parameters['sfh'] = 1 # tau_model but default is constant and can modify the value
        parameters['add_neb_emission'] = False
        parameters['add_neb_continuum'] = False
        parameters['imf_type'] = 1
    elif dataset == 2:
        parameters['dust2'] = jax.random.uniform(next(subkeys), minval=0.0, maxval=2.0)  # uniform distribution between 0 and 1
        parameters['sfh'] = jax.random.choice(next(subkeys), jnp.array([1, 4]))
        parameters['add_neb_emission'] = False
        parameters['add_neb_continuum'] = False
        parameters['dust_type'] = jax.random.choice(next(subkeys), jnp.array([0, 1, 2]))  # 0: no dust, 1: Calzetti, 2: SMC
        parameters['imf_type'] = 1
        # specific to sfh=4
        parameters['tau'] = jax.random.uniform(next(subkeys), minval=0.1, maxval=10.0)  # uniform distribution between 0.1 and 10
    elif dataset == 3:
        parameters['dust2'] = jax.random.uniform(next(subkeys), minval=0.0, maxval=2.0)
        parameters['sfh'] = jax.random.choice(next(subkeys), jnp.array([1, 4]))
        parameters['add_neb_emission'] = True
        parameters['add_neb_continuum'] = True
        parameters['dust_type'] = jax.random.choice(next(subkeys), jnp.array([0, 1, 2]))  # 0: no dust, 1: Calzetti, 2: SMC
        # specific to sfh=4
        parameters['tau'] = jax.random.uniform(next(subkeys), minval=0.1, maxval=10.0)  # uniform distribution between 0.1 and 10
        parameters['fburst'] = jax.random.uniform(next(subkeys), minval=0.01, maxval=1.0)  # uniform distribution between 0 and 1  
        parameters['dust1'] = jax.random.uniform(next(subkeys), minval=0.0, maxval=2.0)  # uniform distribution between 0 and 2
        parameters['gas_logu'] = jax.random.uniform(next(subkeys), minval=-4.0, maxval=-1.0)  # uniform distribution between -4 and -1
        parameters['imf_type'] = jax.random.choice(next(subkeys), jnp.array([0, 1, 2]))  # 0: Salpeter, 1: Chabrier, 2: Kroupa

    # logzsol is the metallicity in solar units
    parameters['logzsol'] = jax.random.uniform(next(subkeys), minval=-2.0, maxval=0.2)  # uniform distribution between -2 and 0.2
    if dataset == 3:
        parameters['gas_logz'] = parameters['logzsol']  # gas metallicity is the same as stellar metallicity

    # tage is the age of the galaxy in Gyr
    parameters['tage'] = jax.random.uniform(next(subkeys), minval=0.01, maxval=13.8)  # uniform distribution between 0.01 and 13.8 Gyr
    if dataset == 3:
        parameters['tburst'] = jax.random.uniform(next(subkeys), minval=0.01, maxval=parameters['tage'])  # uniform distribution between 0.01 and 13.8 Gyr

    # only gets used if sfh=1
    if dataset != 3:
        parameters['const'] = 1.0

    return parameters

N_samples = 50000

# Calculate the number of samples for each rank, distributing the remainder
base = N_samples // size
remainder = N_samples % size
if rank < remainder:
    perrank = base + 1
    start_idx = rank * perrank
else:
    perrank = base
    start_idx = remainder * (base + 1) + (rank - remainder) * base

os.makedirs('fsps-data' + str(dataset), exist_ok=True)

comm.Barrier()  # wait for all processes to be ready    


keys = jax.random.split(key, perrank)
dataset_params = [prior(k) for k in keys]  # each is a dict

results = [fsps_call(p) for i, p in enumerate(tqdm.tqdm(dataset_params, desc="Generating FSPS spectra"))]
lam, spec = zip(*results)
lam = jnp.array(lam)
spec = jnp.array(spec)
print("lam shape:", lam.shape)

for i in range(perrank):
    filename = f"fsps-data" + str(dataset) + f"/spectrum_{start_idx + i}.npz"
    jnp.savez(filename, lam=lam[i], spec=spec[i], **dataset_params[i])
print(f"Rank {rank} saved {perrank} spectra to fsps-data1")
spec = jnp.array(spec)
print("lam shape:", lam.shape)