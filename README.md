# Normative modelling using conditional VAE

Official script for the paper "Conditional VAEs for confound removal and normative modelling of neurodegenerative diseases".

## Abstract 

Understanding pathological mechanisms for heterogeneous brain disorders is a difficult challenge. Normative modelling provides a statistical description of the `normal' range that can be used at subject level to detect deviations, which relate to disease presence, disease severity or disease subtype. Here we trained a conditional Variational Autoencoder (cVAE) on structural MRI data from healthy controls to create a normative model conditioned on confounding variables such as age. The cVAE allows us to use deep learning to identify complex relationships that are independent of these confounds which might otherwise inflate pathological effects. We propose a latent deviation metric and use it to quantify deviations in individual subjects with neurological disorders and, in an independent Alzheimer's disease dataset, subjects with varying degrees of pathological ageing. Our model is able to identify these disease cohorts as deviations from the normal brain in such a way that reflect disease severity. 

## Installation and running scripts

Clone this repository and move to folder:
```bash
git clone https://github.com/alawryaguila/multiAE
cd normativecVAE
```

Create the customised python environment:
```bash
conda create --name norm
```

Activate python environment:
```bash
conda activate norm
```

Install normativecVAE package:
```bash
python setup.py install
```

Example of how to run scripts:
```bash
python train_VAE.py --path /path/to/data --outpath /path/to/save/model
```

## Implementations and Model Settings

To generate the results illustrated in the submission, we trained both VAE and cVAE models using the Adam optimizer with the following model parameters: learning rate=0.001, batch size=1000, latent dimension = 10, size of hidden dense layer = 40 and number of dense layers in each encoder and decoder = 2. We trained our models on a machine with 1 NVIDIA GeForce MX150 GPU.



