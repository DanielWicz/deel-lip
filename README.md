<div align="center">
        <picture>
                <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/banner_dark_deellip.png">
                <source media="(prefers-color-scheme: light)" srcset="./docs/assets/banner_light_deellip.png">
                <img alt="DEEL-LIP Banner" src="./docs/assets/banner_light_deellip.png">
        </picture>
</div>
<br>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/pypi/pyversions/deel-lip.svg">
    </a>
    <a href="https://github.com/deel-ai/deel-lip/actions/workflows/python-linters.yml">
        <img alt="PyLint" src="https://github.com/deel-ai/deel-lip/actions/workflows/python-linters.yml/badge.svg?branch=master">
    </a>
    <a href="https://github.com/deel-ai/deel-lip/actions/workflows/python-tests.yml">
        <img alt="Tox" src="https://github.com/deel-ai/deel-lip/actions/workflows/python-linters.yml/badge.svg?branch=master">
    </a>
    <a href="https://pypi.org/project/deel-lip">
        <img alt="Pypi" src="https://img.shields.io/pypi/v/deel-lip.svg">
    </a>
    <a href="https://pepy.tech/project/deel-lip">
        <img alt="Pepy" src="https://pepy.tech/badge/deel-lip">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
    <br>
    <a href="https://deel-ai.github.io/deel-lip/"><strong>Explore DEEL-LIP docs »</strong></a>
</div>
<br>

## 👋 Welcome to deel-lip documentation!

Controlling the Lipschitz constant of a layer or a whole neural network
has many applications ranging from adversarial robustness to Wasserstein
distance estimation.

This library provides an efficient implementation of **k-Lipschitz
modules for PyTorch**.

> [!NOTE]
> Looking for the historical TensorFlow/Keras implementation?
> Check out the `keras3` branch for the legacy API and migration notes.


## 📚 Table of contents

- [📚 Table of contents](#-table-of-contents)
- [🔥 Tutorials](#-tutorials)
- [🚀 Quick Start](#-quick-start)
- [📦 What's Included](#-whats-included)
- [👍 Contributing](#-contributing)
- [👀 See Also](#-see-also)
- [🙏 Acknowledgments](#-acknowledgments)
- [🗞️ Citation](#-citation)
- [📝 License](#-license)

## 🚀 Quick Start

You can install ``deel-lip`` directly from PyPI:

```bash
pip install deel-lip
```

In order to use ``deel-lip``, you also need a [valid PyTorch
installation](https://pytorch.org/get-started/locally/). ``deel-lip``
supports PyTorch versions 2.x.

```python
import torch
from deel.lip.layers import SpectralLinear

layer = SpectralLinear(32, 16, bias=False)  # Lipschitz constrained torch.nn.Linear
x = torch.randn(4, 32)
out = layer(x)
print(out.shape)  # torch.Size([4, 16])
```

## 🔥 Tutorials

| **Tutorial Name**           | Notebook                                                                                                                                                           |
| :-------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Getting Started 1 - Building a Lipschitz MLP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/docs/notebooks/Getting_started_1.ipynb)            |
| Getting Started 2 - Training a Lipschitz network | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/docs/notebooks/Getting_started_2.ipynb)            |
| Demo 0 - Inspecting spectral normalization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/docs/notebooks/demo0.ipynb) |
| Demo 1 - Computing provable robustness metrics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/docs/notebooks/demo1.ipynb) |
| Demo 2 - Training with HKR loss | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/docs/notebooks/demo2.ipynb) |
| Demo 3 - Exporting condensed models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/docs/notebooks/demo3.ipynb) |
| Demo 4 - Monitoring Lipschitz constraints during training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/deel-lip/blob/master/docs/notebooks/demo4.ipynb) |


## 📦 What's Included

*  k-Lipschitz variants of `torch.nn` layers such as ``Linear`` and ``Conv2d``,
*  activation functions implemented with ``torch.nn`` modules,
*  kernel initializers and kernel constraints for PyTorch parameters,
*  loss functions that make use of Lipschitz constrained networks (see
   [our paper](https://arxiv.org/abs/2006.06520) for more
   information),
*  tools to monitor the singular values of kernels during training,
*  tools to convert k-Lipschitz network to regular network for faster
   inference.

## 👍 Contributing

PyTorch implementation, Daniel Wiczew, NEBULA

## 👀 See Also

More from the DEEL project:

- [Xplique](https://github.com/deel-ai/xplique) a Python library exclusively dedicated to explaining neural networks.
- [Influenciae](https://github.com/deel-ai/influenciae) Python toolkit dedicated to computing influence values for the discovery of potentially problematic samples in a dataset.
- [deel-TorchLip](https://github.com/deel-ai/deel-torchlip) a Python library for training k-Lipschitz neural networks on PyTorch.
- [Oodeel](https://github.com/deel-ai/oodeel) a Python library for post-hoc deep OOD (Out-of-Distribution) detection on already trained neural network image classifiers
- [DEEL White paper](https://arxiv.org/abs/2103.10529) a summary of the DEEL team on the challenges of certifiable AI and the role of data quality, representativity and explainability for this purpose.

## 🙏 Acknowledgments

<div align="right">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://share.deel.ai/apps/theming/image/logo?useSvg=1&v=10"  width="25%" align="right">
    <source media="(prefers-color-scheme: light)" srcset="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png"  width="25%" align="right">
    <img alt="DEEL Logo" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png" width="25%" align="right">
  </picture>
</div>
This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.

## 🗞️ Citation

This library has been built to support the work presented in the paper
[Achieving robustness in classification using optimaltransport with
Hinge regularization](https://arxiv.org/abs/2006.06520) which aim
provable and efficient robustness by design.

This work can be cited as:

```
@misc{2006.06520,
    Author = {Mathieu Serrurier and Franck Mamalet and Alberto González-Sanz and Thibaut Boissin and Jean-Michel Loubes and Eustasio del Barrio},
    Title = {Achieving robustness in classification using optimal transport with hinge regularization},
    Year = {2020},
    Eprint = {arXiv:2006.06520},
}
```

## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.
