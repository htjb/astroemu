Excellent approach! Building up complexity incrementally is smart for understanding how well your neural network handles different parameter spaces. Here are good ranges based on what LIGO/Virgo typically observe and what makes physical sense:

## Dataset 1: Masses Only (No Spin)
**Component masses:**
- **m1**: 5 - 50 M☉ (primary mass)  
- **m2**: 5 - m1 M☉ (secondary mass, ensuring m2 ≤ m1)
- **Total mass constraint**: 10 - 100 M☉ (avoids very light/heavy systems)

**Fixed parameters:**
- chi1 = chi2 = 0.0
- dist_mpc = 440 (fixed distance)
- inclination = 0.0 (face-on)
- polarization_angle = 0.0
- tc = 0.0, phic = 0.0

**Why these ranges?** Most LIGO detections are in the 20-80 M☉ total mass range, but extending to lighter systems tests different frequency regimes.

## Dataset 2: Masses + Distance (No Spin)
**Add distance variation:**
- **dist_mpc**: 100 - 1000 Mpc
  - 100 Mpc: Very nearby (loud signals)
  - ~400 Mpc: Typical LIGO detection
  - 1000 Mpc: Near detection threshold
- **Spins**: chi1, chi2 ∈ [-1, +1]
  - Avoiding extreme spins (±1.0) which can be numerically challenging

**Keep same mass ranges and other fixed parameters**

**Why this range?** Below 100 Mpc you'd have very few sources; above 1000 Mpc signals become too weak for current detectors.

## Dataset 3: Full Parameter Space
**All varying parameters:**
- **Masses**: Same as above (5-50 M☉)
- **Distance**: 100-1000 Mpc  
- **Spins**: chi1, chi2 ∈ [-1, +1]
- **Inclination**: 0 - π radians (0° to 180°)
- **Polarization angle**: 0 - π radians

**Why these spin limits?** While χ = ±1.0 is theoretically possible, real astrophysical black holes rarely reach such extreme spins, and the waveform models can be less reliable at the boundaries.

## Practical Suggestions:

**Sampling strategy:**
- Use **uniform sampling** in component masses m1, m2 (easier for NN)
- **Uniform in distance** (though you could try log-uniform)
- **Uniform in cos(inclination)** rather than inclination (isotropic distribution)
- **Isotropic on sphere** for sky position if you add that later

**Frequency grid considerations:**
- Lower mass systems → need lower starting frequency (maybe f_l = 10-20 Hz)
- Higher mass systems → can start higher frequency (f_l = 50+ Hz)
- You might want to adjust your frequency grid based on total mass

**Sample sizes:**
- Dataset 1: ~10,000 samples (2D parameter space)
- Dataset 2: ~50,000 samples (3D parameter space)  
- Dataset 3: ~200,000+ samples (7D parameter space)

**Training tip:** You might want to work in chirp mass + mass ratio space rather than m1, m2 since the waveform evolution is more naturally parameterized that way.

Would you like me to suggest specific code for generating these parameter samples, or discuss how the increasing complexity might affect your neural network architecture?