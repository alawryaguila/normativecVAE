from argparse import ArgumentParser
import pandas as pd
from os.path import join
import numpy as np
pd.set_option('display.float_format', str)

def process():
    parser = ArgumentParser(description="Train VAE")
    parser.add_argument('--path', type=str, default='', help='Path to input data')
    args = parser.parse_args()
    controls_data = pd.read_csv(join(args.path, 'train_controls_220215.csv'),header=0)
    to_drop = ['CSF', 'Optic.Chiasm',  'X3rd.Ventricle',	'X4th.Ventricle',	'X5th.Ventricle',
            'Left.Cerebellum.Cortex', 'Right.Cerebellum.Cortex', 'Left.Inf.Lat.Vent', 'Right.Inf.Lat.Vent', 'Right.VentralDC',
            'Left.VentralDC',
        'Left.Cerebellum.White.Matter', 'Right.Cerebellum.White.Matter',  'Left.vessel', 'Right.vessel', 'Brain.Stem']

    controls_data = controls_data[controls_data.columns.drop(to_drop)]
    controls_data.reset_index(inplace=True, drop=True)
    controls_data = controls_data.rename(columns={'Right.Lateral.Ventricle':'RLatVent', 'Left.Lateral.Ventricle':'LLatVent'})
    controls_data.reset_index(inplace=True, drop=True)

    non_data_cols = ['eid', 'AGE', 'SEX', 'ICV']
    controls_data = controls_data[controls_data['AGE'].notna()]
    controls_data = controls_data[controls_data['SEX'].notna()]
    controls_data = controls_data[controls_data['ICV'].notna()]
    controls_data = controls_data.dropna()
    
    #restrict controls data to between 47-73 years
    print(controls_data.shape)

    controls_data = controls_data[controls_data.AGE>=47]
    controls_data = controls_data[controls_data.AGE<=73]
    controls_data = controls_data.groupby('SEX').apply(lambda x: x.sample(n=controls_data[controls_data['SEX'] == 1].shape[0], random_state=42))
    controls_data.reset_index(inplace=True, drop=True)
    print(controls_data.shape)
    controls_data = controls_data.dropna()
    controls_data.reset_index(inplace=True, drop=True)
    print(controls_data.head())
    controls_labels = controls_data[non_data_cols]
    controls_data = controls_data.drop(non_data_cols, axis=1)
    
    controls_cols = controls_data.columns
    
    data = pd.read_csv(join(args.path,'test_ms_data_220215.csv'), header=0)

    data_2 = pd.read_csv(join(args.path,'test_mania_data_220215.csv'), header=0)
    data_2 = data_2[data_2.DX == 1]
    data = pd.concat([data, data_2])
    #Drop subjects where APOE4 is NaN
    data = data[data['DX'].notna()]
    data = data[data['AGE'].notna()]
    data = data[data['ICV'].notna()]
    data = data[data['SEX'].notna()]
    data = data.rename(columns={'Right.Lateral.Ventricle':'RLatVent', 'Left.Lateral.Ventricle':'LLatVent'})
    data = data.dropna()

    data = data[data.AGE>=47]
    data = data[data.AGE<=73]
    data = data[data.ICV>=controls_labels.ICV.min()]
    data = data[data.ICV<=controls_labels.ICV.max()]
    data.reset_index(inplace=True, drop=True)

    non_data_cols = ['eid', 'AGE', 'SEX', 'ICV', 'DX']
    labels = data[non_data_cols]
    data = data.drop(non_data_cols, axis=1)
    data = data[controls_cols]

    covariates = labels[[ 'AGE', 'ICV', 'SEX', 'DX']] #leave site out for now

    controls_covariates = controls_labels[['AGE', 'ICV', 'SEX']]

    #center controls data

    mean_controls = np.mean(controls_data, axis=0)
    sd_controls = np.std(controls_data, axis=0)
    controls_data = (controls_data - mean_controls)/sd_controls
    data = (data - mean_controls)/sd_controls
    
    data.to_csv('../data/test_data.csv', index=False)
    covariates.to_csv('../data/test_covariates.csv', index=False)
    controls_data.to_csv('../data/train_data.csv', index=False)
    controls_covariates.to_csv('../data/train_covariates.csv', index=False)
    #convert age labels into buckets
    bin_labels = list(range(0,10))
    age_bins_controls, bin_edges = pd.qcut(controls_covariates['AGE'], 10, retbins=True, labels=bin_labels)
    age_bins_cases = pd.cut(covariates['AGE'], bins=bin_edges, labels=bin_labels)
    age_bins_cases = np.eye(10)[age_bins_cases.values]
    age_bins_controls = np.eye(10)[age_bins_controls.values]

    ICV_bins_controls, bin_edges = pd.qcut(controls_covariates['ICV'], q=10, retbins=True, labels=bin_labels)
    one_hot_ICV_controls = np.eye(10)[ICV_bins_controls.values]
    #make sure bin edges include edges of age range too 
    ICV_bins_cases = pd.cut(covariates['ICV'], bins=bin_edges, labels=bin_labels)
    one_hot_ICV = np.eye(10)[ICV_bins_cases.values]

if __name__ == '__main__':
    process()