# VISTA-SSM: Varying and Irregular Sampling Time-series Analysis via State Space Models

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/benjaminbrindle/vista_ssm/blob/main/LICENSE)

We introduce VISTA, the Python implementation of the paper _VISTA-SSM: Varying and Irregular Sampling Time-series Analysis via State Space Models_. Real world time series data, particularly in the psychological sciences, is often characterized by irregular or ill-structured intervals. This challenge often emerges when exploring temporal data without clear prior categorical information about the subjects in a population. VISTA provides a new method for handling these sampling issues while performing unsupervised identification of groups (clustering) in such datasets. Our approach adapts linear Gaussian state space models (LGSSMs) to provide a flexible parametric framework for fitting a wide range of time series dynamics. The clustering approach itself is based on the assumption that the population can be represented as a mixture of a given number of LGSSMs. The algorithmic structure of VISTA is sketched in the below schematic, with readers encouraged to reference our paper for more technical details and derivations. We hope that researchers will find VISTA to be an accessible and valuable tool for tackling some of the challenges found in healthcare and mental health data.

![VISTA Schematic](https://github.com/benjaminbrindle/vista_ssm/blob/main/paper_schematic.jpg)

## Installation 
```bash
git clone git@github.com:benjaminbrindle/vista_ssm.git
pip install -e vista_ssm
```

## Usage

Examples can be found in our notebook files.
