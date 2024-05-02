import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt; plt.style.use("dark_background")
import seaborn as sns
import sys
import sys; sys.path.append("/data/jerrylee/pjt/BIGFAM.v.0.1")
import statsmodels.formula.api as smf
import statsmodels.api as sm
from BIGFAM import obj2, tools, frreg
import importlib

source = "UKB" # UKB, GS

frreg_path = f"/data/jerrylee/pjt/BIGFAM.v.0.1/data/{source}/frreg/REL"
pheno_fns = os.listdir(frreg_path)


df_bigfam = pd.DataFrame(
    columns=["pheno", "weight", "X_BIGFAM", "lower_BIGFAM", "upper_BIGFAM"]
)

for pheno_fn in tqdm(pheno_fns):
    pheno = pheno_fn.split(".")[0]
    df_frreg = pd.read_csv(
        f"{frreg_path}/{pheno_fn}",
        sep='\t'
    )
    
    # regress out mean
    for alp in range(-3, 10):
        result = obj2.estimateX(df_frreg, 
                                alpha_dicts={"type": "eta", "weight":alp})
        
        df_bigfam.loc[len(df_bigfam)] = [pheno, alp, 
                                         np.median(result["X"]),
                                         np.percentile(result["X"], 2.5), 
                                         np.percentile(result["X"], 97.5)]

df_bigfam.to_csv("/data/jerrylee/pjt/BIGFAM.v.0.1/test/varing_weight.tsv",
                 sep='\t',
                 index=False)