from setuptools import setup
from pathlib import Path

VERSION = "0.0.1"

with Path('requirements.txt').open() as f:
    INSTALL_REQUIRES = [line.strip() for line in f.readlines() if line]

setup(
    name = 'vista_ssm',
    author = 'Benjamin Brindle',
    author_email = 'brindlebenjamin@gmail.com',
    url = 'https://github.com/benjaminbrindle/vista_ssm',
    description = 'VISTA-SSM: Varying and Irregular Sampling Time-series Analysis via State Space Models',
    version = VERSION,
    packages = ['vista_ssm'],
    install_requires = INSTALL_REQUIRES
)