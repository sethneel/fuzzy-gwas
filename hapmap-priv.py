import torch
from sklearn.decomposition import PCA
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

# please set the path to your data directory here
data_path = '/Users/sneel/Dropbox/Research/Current_Projects/epi-data/Data'
from gen_helpers import *
# read in genotype data from CEU population (112 respondents, 718848 snps)
geno_matrix = read_geno(pname('CEU.geno', data_path))

# remove SNPs with missing data: 500k by 112
geno_matrix = np.ma.compress_rows(geno_matrix)

# take a random subset of SNPs and compute a phenotype


# goal privately compute the allele frequency:
# 1. Learn low dimensional representation (Auto-Encoder?)
# 2. Compute average loadings
# 3. Convert back to average in high dimensional space
# 4. Assess accuracy

# compute principle components
# transpose
svd = np.linalg.svd(geno_matrix, full_matrices=False, compute_uv=True)
U = svd[0]
s = svd[1]
Vh = svd[2]

k_components = 1

U_k = U[:, range(k_components)]
S_k = np.diag(s[range(k_components)])
V_kh = Vh[range(k_components), :]
G_k = np.matmul(np.matmul(U_k, S_k), V_kh)
mu_k = np.mean(G_k, axis=0)
mu = np.mean(geno_matrix, axis=0)
# compute the mean absolute reconstruction error of the mean
MAE = np.mean(np.abs(mu_k-mu))