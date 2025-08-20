import jax.numpy as jnp
import jax
from ripplegw.waveforms import IMRPhenomD
from ripplegw import ms_to_Mc_eta
import os
from mpi4py import MPI as mpi

rank = mpi.COMM_WORLD.Get_rank()
size = mpi.COMM_WORLD.Get_size()
comm = mpi.COMM_WORLD

key = jax.random.PRNGKey(rank)

@jax.jit
def generate_waveform(params, fs, f_ref):
    # chirp mass (m1*m2)^(3/5) / (m1 + m2)^(1/5) - determines ealy inspiral evolution
    # symmetric mass ratio eta = m1*m2 / (m1 + m2)^2 - determins merger dynamics
    Mc, eta = ms_to_Mc_eta(jnp.array([params['m1'], params['m2']]))

    # These are the parametrs that go into the waveform generator
    # Note that JAX does not give index errors, so if you pass in the
    # the wrong array it will behave strangely
    theta_ripple = jnp.array(
        [Mc, eta, params['chi1'], params['chi2'], 
         params['dist_mpc'], params['tc'], params['phic'], 
         params['inclination'], params['polarization_angle']]
    )

    # And finally lets generate the waveform!
    hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_hphc(fs, theta_ripple, f_ref)
    return [hp_ripple, hc_ripple]

def prior(key, data_set_id):

    subkeys = iter(jax.random.split(key, 100))

    parameters = {}
    if data_set_id == 1:
        parameters['m1'] = jax.random.uniform(next(subkeys), minval=5.0, maxval=50.0) 
        parameters['m2'] = jax.random.uniform(next(subkeys), minval=5.0, maxval=parameters["m1"])
        parameters['chi1'] = 0.0
        parameters['chi2'] = 0.0
        parameters['dist_mpc'] = 440.0
        parameters['tc'] = 0.0
        parameters['phic'] = 0.0
        parameters['inclination'] = 0.0
        parameters['polarization_angle'] = 0.0
    
    elif data_set_id == 2:
        parameters['m1'] = jax.random.uniform(next(subkeys), minval=5.0, maxval=50.0) 
        parameters['m2'] = jax.random.uniform(next(subkeys), minval=5.0, maxval=parameters["m1"])
        parameters['chi1'] = jax.random.uniform(next(subkeys), minval=-1.0, maxval=1.0)
        parameters['chi2'] = jax.random.uniform(next(subkeys), minval=-1.0, maxval=1.0)
        parameters['dist_mpc'] = jax.random.uniform(next(subkeys), minval=100.0, maxval=1000.0)
        parameters['tc'] = 0.0
        parameters['phic'] = 0.0
        parameters['inclination'] = 0.0
        parameters['polarization_angle'] = 0.2

    elif data_set_id == 3:
        parameters['m1'] = jax.random.uniform(next(subkeys), minval=5.0, maxval=50.0) 
        parameters['m2'] = jax.random.uniform(next(subkeys), minval=5.0, maxval=parameters["m1"])
        parameters['chi1'] = jax.random.uniform(next(subkeys), minval=-1.0, maxval=1.0)
        parameters['chi2'] = jax.random.uniform(next(subkeys), minval=-1.0, maxval=1.0)
        parameters['dist_mpc'] = jax.random.uniform(next(subkeys), minval=100.0, maxval=1000.0)
        parameters['tc'] = 0.0
        parameters['phic'] = 0.0
        parameters['inclination'] = jax.random.uniform(next(subkeys), minval=0.0, maxval=jnp.pi)
        parameters['polarization_angle'] = jax.random.uniform(next(subkeys), minval=0.0, maxval=jnp.pi)
    
    return parameters

N_samples = 500000

# Now we need to generate the frequency grid
f_l = 24
f_u = 512
del_f = 0.01
fs = jnp.arange(f_l, f_u, del_f)

# We also need to give a reference frequency
f_ref = f_l

# Calculate the number of samples for each rank, distributing the remainder
base = N_samples // size
remainder = N_samples % size
if rank < remainder:
    perrank = base + 1
    start_idx = rank * perrank
else:
    perrank = base
    start_idx = remainder * (base + 1) + (rank - remainder) * base

dataset = 3  # Example dataset ID, can be changed as needed

os.makedirs('ripple-IMRPhenomD-data' + str(dataset), exist_ok=True)

comm.Barrier()  # wait for all processes to be ready    

keys = jax.random.split(key, perrank)
dataset_params = [prior(k, dataset) for k in keys]  # each is a dict
waveforms = [generate_waveform(p, fs, f_ref) for p in dataset_params]

for i in range(perrank):
    hp_ripple, hc_ripple = waveforms[i]
    print(hp_ripple, hc_ripple)  # Debugging output, can be removed later
    jnp.savez(f'ripple-IMRPhenomD-data{dataset}/waveform_{start_idx + i}.npz',
             fs=fs, hp_ripple_real=hp_ripple.real, hc_ripple_real=hc_ripple.real, 
             hp_ripple_imag=hp_ripple.imag, hc_ripple_imag=hc_ripple.imag,
             **dataset_params[i])
