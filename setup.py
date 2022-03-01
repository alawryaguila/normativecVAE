from setuptools import setup, find_packages
import os

def setup_package():
    data = dict(
    name='normativecVAE', 
    packages=find_packages(),
    install_requires=[
        'statsmodels==0.10.1',
        'umap_learn==0.5.2',
        'matplotlib==3.4.1',
        'numpy==1.19.5',
        'torch==1.1.0',
        'setuptools==54.2.0',
        'pandas==1.2.3',
        'scikit_learn==1.0.2',
        'umap==0.1.1',
    ],
    description='A library for running cVAE normative models'
    )
    setup(**data)

if __name__ == "__main__":
    setup_package()