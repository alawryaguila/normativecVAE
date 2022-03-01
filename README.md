# Normative modelling using conditional VAE

Official script for the paper "Conditional VAEs for confound removal and normative modelling of neurodegenerative diseases".

## Abstract 

Understanding pathological mechanisms for heterogeneous brain disorders is a difficult challenge. Normative modelling provides a statistical description of the `normal' range that can be used at subject level to detect deviations, which relate to disease presence, disease severity or disease subtype. Here we trained a conditional Variational Autoencoder (cVAE) on structural MRI data from healthy controls to create a normative model conditioned on confounding variables such as age. The cVAE allows us to use deep learning to identify complex relationships that are independent of these confounds which might otherwise inflate pathological effects. We propose a latent deviation metric and use it to quantify deviations in individual subjects with neurological disorders and, in an independent Alzheimer's disease dataset, subjects with varying degrees of pathological ageing. Our model is able to identify these disease cohorts as deviations from the normal brain in such a way that reflect disease severity. 
