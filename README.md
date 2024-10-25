# VISTA-SSM: Varying and Irregular Sampling Time-series Analysis via State Space Models

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/benjaminbrindle/vista_ssm/blob/main/LICENSE)

We introduce VISTA, a Python tool based on the method described in _VISTA-SSM: Varying and Irregular Sampling Time-series Analysis via State Space Models_. 

Real world time series data, particularly in the psychological sciences, is often characterized by irregular or ill-structured intervals. This challenge often emerges when exploring temporal data without clear prior categorical information about the subjects in a population. 

VISTA provides a new method for handling these sampling issues while performing unsupervised identification of groups (clustering) in such datasets. Our approach adapts linear Gaussian state space models (LGSSMs) to provide a flexible parametric framework for fitting a wide range of time series dynamics. The clustering approach itself is based on the assumption that the population can be represented as a mixture of a given number of LGSSMs. The algorithmic structure of VISTA is sketched in the below schematic, with readers encouraged to reference our paper for more technical details and derivations.

We hope that VISTA will be an accessible and valuable tool for researchers to handle the challenges of healthcare and mental health data.


![VISTA Schematic](https://github.com/benjaminbrindle/vista_ssm/blob/main/schematic.jpg)

## Installation 
```bash
git clone git@github.com:benjaminbrindle/vista_ssm.git
pip install -e vista_ssm
```

## Usage

We have compiled a number of useful helper functions to interface directly with the VISTA algorithm, which was inspired by (Umatani et al., 2023)[^1]. For instance, one can run the VISTA algorithm by simply running the function:
```
runVISTA(how,param_dic,dataset,time_points,**kwargs):
    """helper function to run EMLGSSM algorithm

    Parameters
    ----------
    how: string
        which method to use for initializing the parameters:
            random: chooses random parameters within a specific range
            ident: chooses parameters as identity matrices or close to them
            kmeans: uses kmeans on the data to choose parameters
    param_dic: dic
        parameters used:
            DIM_X
            DIM_Y
            N_CLUSTER
            NUM_CPU
            FIX - list of parameters to be fixed throughout algorithm
            NUM_LGSSM - number of lgssms to use in kmeans initialization for each time series
            MAX_ITER
            EPSILON
            BIC - bool of whether or not to return bayesian information criterion
            
    dataset: ndarray(n_samples,n_time,dim_y,1)

    time_points: ndarray(n_samples,n_time)
        times corresponding to each observation in dataset
"""
```
Within the experiments folder we provide several Jupyter notebooks showcasing VISTA and demonstrating its ease of use in practical settings. In the data folder we have collected the open-source datasets used in our panel[^2], epidemiological[^3], and ecological momentary assessment[^4] data examples for ease of reproducibility. We have compiled our results from running VISTA on each of these datasets in the results folder.

[^1]: [Umatani, R., Imai, T., Kawamoto, K., & Kunimasa, S. (2023). Time series clustering with
an em algorithm for mixtures of linear gaussian state space models. Pattern
Recognition, 138, 109375.](https://github.com/ur17/em_mlgssm)

[^2]: [World Health Organization COVID-19 Dashboard](https://data.who.int/dashboards/covid19)

[^3]: [U.S. Census Bureau Historical Annual Time Series of State Population Estimates](https://web.archive.org/web/20040220002039/https://eire.census.gov/popest/archives/state/st_stts.php)

[^4]: [Fried, E. I., Papanikolaou, F., & Epskamp, S. (2022). Mental health and social contact
during the covid-19 pandemic: An ecological momentary assessment study. Clinical
Psychological Science, 10 (2), 340–354.](https://osf.io/erp7v/files/osfstorage)
