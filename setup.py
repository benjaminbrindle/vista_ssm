from setuptools import setup, find_packages
from pathlib import Path

VERSION = "1.0.0"

with Path('requirements.txt').open() as f:
    INSTALL_REQUIRES = [line.strip() for line in f.readlines() if line]

setup(
    name = 'vista_ssm',
    author = 'Benjamin Brindle',
    author_email = 'brindlebenjamin@gmail.com',
    url = 'https://github.com/benjaminbrindle/vista_ssm',
    description = 'VISTA-SSM: Varying and Irregular Sampling Time-series Analysis via State Space Models',
    version = VERSION,
    packages = find_packages(include=['vista_ssm', 'vista_ssm.*']),
    install_requires = INSTALL_REQUIRES
)