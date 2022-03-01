import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
from os.path import join
import pandas as pd
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LinearRegression
import statsmodels as sm
from statsmodels.api import OLS
class MyDataset_labels(Dataset):
    def __init__(self, data, labels, indices = False, transform=None):
        self.data = data
        self.labels = labels
        if isinstance(data,list) or isinstance(data,tuple):
            self.data = [torch.from_numpy(d).float() if isinstance(d,np.ndarray) else d for d in self.data]
            self.N = len(self.data[0])
            self.shape = np.shape(self.data[0])
        elif isinstance(data,np.ndarray):
            self.data = torch.from_numpy(self.data).float()
            self.N = len(self.data)
            self.shape = np.shape(self.data)
            
        self.labels = torch.from_numpy(self.labels).long()

        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        if isinstance(self.data,list):
            x = [d[index] for d in self.data]
        else:
            x = self.data[index]

        if self.transform:
            x = self.transform(x)
        t = self.labels[index]
        if self.indices:
            return x, t, index
        return x, t

    def __len__(self):
        return self.N

class MyDataset(Dataset):
    def __init__(self, data, indices = False, transform=None):
        self.data = data
        if isinstance(data,list) or isinstance(data,tuple):
            self.data = [torch.from_numpy(d).float() if isinstance(d,np.ndarray) else d for d in self.data]
            self.N = len(self.data[0])
            self.shape = np.shape(self.data[0])
        else:
            if isinstance(data,np.ndarray):
                self.data = torch.from_numpy(self.data).float()
            self.N = len(self.data)
            self.shape = np.shape(self.data)

        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        if isinstance(self.data,list):
            x = [d[index] for d in self.data]
        else:
            x = self.data[index]

        if self.transform:
            x = self.transform(x)

        if self.indices:
            return x, index
        return x

    def __len__(self):
        return self.N
        
def plot_losses(logger, path, title=''):
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Loss values')
    for k, v in logger.logs.items(): 
        plt.plot(v, label=str(k))
    plt.xlabel('epochs', fontsize=10)
    plt.ylabel('loss', fontsize=10)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Loss relative values')
    for k, v in logger.logs.items(): 
        max_loss = 1e-8 + np.max(np.abs(v))
        plt.plot(v/max_loss, label=str(k))
    plt.legend()
    plt.xlabel('epochs', fontsize=10)
    plt.ylabel('loss', fontsize=10)
    plt.savefig(join(path, "Losses{0}.png".format(title)))
    plt.close()

class Logger():
    
    def __init__(self):
        self.logs = {}

    def on_train_init(self, keys):
        for k in keys:
            self.logs[k] = []

    def on_step_fi(self,logs_dict):
        for k,v in logs_dict.items():
            self.logs[k].append(v.detach().cpu().numpy())

def reconstruction_deviation(x, x_pred):
    return np.sum((x - x_pred)**2, axis=1)/x.shape[1]

def latent_deviation(mu_train, mu_sample, var_sample):
    var = np.var(mu_train, axis=0)
    return np.sum(np.abs(mu_sample - np.mean(mu_train, axis=0))/np.sqrt(var + var_sample), axis=1)/mu_sample.shape[1]

def separate_latent_deviation(mu_train, mu_sample, var_sample):
    var = np.var(mu_train, axis=0)
    return (mu_sample - np.mean(mu_train, axis=0))/np.sqrt(var + var_sample)

def latent_pvalues(latent, target, type):
    pval_df = pd.DataFrame({'labels': ['const', 'latent']})
    for i in range(latent.shape[1]):
        latent_curr = latent[:,i]
        latent_curr = sm.tools.tools.add_constant(latent_curr)
        if type == 'continuous':
            model = OLS(target, latent_curr)
        else:
            model = Logit(target, latent_curr)
        model_fit = model.fit()
        pval_df['latent {0}'.format(i)] = list(model_fit.pvalues.values)
    return pval_df
