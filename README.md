[![License](https://img.shields.io/pypi/l/aemotrics.svg?color=green)](https://github.com/SurgicalPhotonics/Aemotrics/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/aemotrics.svg?color=green)](https://pypi.org/project/aemotrics)
[![tests](https://github.com/SurgicalPhotonics/Aemotrics/actions/workflows/test.yml/badge.svg)](https://github.com/SurgicalPhotonics/Aemotrics/actions/workflows/test.yml)
[![Docker Image](https://github.com/SurgicalPhotonics/Aemotrics/actions/workflows/docker-build.yml/badge.svg)](https://github.com/SurgicalPhotonics/Aemotrics/actions/workflows/docker-build.yml)
[![Code Style Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# Aemotrics

Unilateral facial palsey quantification using DeepLabCut markerless point tracking.

## Usage
### Local Usage
clone and install aemotrics
```bash
git clone git@github.com:SurgicalPhotonics/aemotrics.git
cd aemotrics
python3 -m build -w
python3 -m pip install dist/*
```
run aemotrics
```bash
python3 -m aemotrics
```
### Cloud Usage
There is a cloud inference version of aemotrics available 
