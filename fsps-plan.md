# FSPS Emulator Training Set Plan

This document outlines a three-tiered structure for generating FSPS-based training data of increasing complexity, to support neural network emulator development.

---

## Dataset 1: Simple (Basic Stellar Populations)

**Purpose:** Test core emulator performance on clean stellar continua.

- `sfh`: 1 (constant SFH)
- `logzsol`: [-1.5, 0.3]
- `tage`: [0.01, 13.0] Gyr
- `const`: 1.0 (if sfh=1)
- `dust2`: 0.0 (no dust)
- `add_neb_emission`: False
- `add_neb_continuum`: False
- `IMF type`: [1] (Chabrier)

---

## Dataset 2: Intermediate (Dust and Star Formation Variability)

**Purpose:** Introduce realistic SED complexity with moderate degeneracies.

- Inherits all from Dataset 1
- `sfh`: 1 or 4 (exponential)
- `tau`: [0.1, 10.0] Gyr (if sfh=4)
- `dust2`: [0.0, 2.0]
- `dust_type`: [0, 1, 2] (MW, SMC, Calzetti)
- `imf_type`: [1] (Chabrier)
- `add_neb_emission`: False
- `add_neb_continuum`: False

---

## Dataset 3: Advanced (Bursts, Nebular, Full Dust Model)

**Purpose:** Full physical realism with sharp features and complex SED behavior.

- Inherits all from Dataset 2
- `fburst`: [0.0, 1.0]
- `tburst`: [0.01, tage]
- `dust1`: [0.0, 2.0]
- `add_neb_emission`: True
- `add_neb_continuum`: True
- `gas_logu`: [-4.0, -1.0]
- `gas_logz`: match `logzsol`
- `imf_type`: [0 1 2] (Salpeter Chabrier Kroupa)

---

## Notes

- All spectra are generated in the rest frame.
- Nebular effects only impact Dataset 3.
- Consistency checks recommended to avoid nonphysical combinations.