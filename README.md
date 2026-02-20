# Next Generation Emulators for Cosmology and Astrophysics

| astroemu | Next Generation Emulators |
|---| ---|
| Author| Harry Bevins|
| Version| 0.2.0 |
| Homepage | https://github.com/htjb/astroemu |

UNDER DEVELOPMENT

`astroemu` implements a generalized framework for emulating 
spectral signals
and is inspired by the [`globalemu`](https://github.com/htjb/globalemu) package.

You can pip install the emulator framework with

```bash
pip install astroemu
```

although the code is in active development.

The neural network emulators are implemented in JAX and the dataloaders are
built on top of PyTorch.

As with `globalemu` the idea is to input the independent variables alongside 
the physical parameters of your model then predicting a single corresponding 
spectral value. Full spectra can then be generated via a vectorised call to 
the network. The training data is tiled in the dataloaders so that the 
parameters and independent variables are concatenated as inputs and
stacked up alongside the outputs. For example if we have a signal 
$y = f(x, \theta)$ and we have N $\theta$ samples and m $x$ and $y$ values then
our training data looks like

|Input|Output|
|--|--|
|[$\theta_{0}$, $x_0$]| $y_0$ |
|[$\theta_{0}$, $x_1$]| $y_1$ |
|[$\theta_{0}$, $x_2$]| $y_2$ |
|[$\theta_{0}$, ...]|...|
|[$\theta_{0}$, $x_m$]|$y_m$|
|[..., ...]| ...|
|[$\theta_N$, $x_m$]|$y_m$|

For more details see the `globalemu` [paper](https://arxiv.org/abs/2104.04336).
A paper is in preparation demonstrating applications of this package to a broad 
range of astrophysical signals.

## Documentation

Docs can be compiled locally with

```
git clone git@github.com:htjb/astroemu.git
pip install ".[docs]"
mkdocs serve
```

or found on [readthedocs](https://astroemu.readthedocs.io/en/latest/)

## Contributions

Contributions are welcome! Please open an issue to discuss and have a 
read of the Contribution guidelines.