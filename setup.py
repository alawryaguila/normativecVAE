from setuptools import setup, find_packages
import os

def setup_package():
    data = dict(
    name='normativecVAE', 
    packages=find_packages(),
    install_requires=[
        'statsmodels==0.13.2',
        'matplotlib==3.5.3',
        'numpy==1.23.2',
        'torch==1.12.1',
        'setuptools==59.5.0',
        'pandas==1.4.3',
    ],
    description='A library for running cVAE normative models'
    )
    setup(**data)

if __name__ == "__main__":
    setup_package()