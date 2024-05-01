# %%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys

import sys; sys.path.append("/data/jerrylee/pjt/BIGFAM.v.0.1")
from BIGFAM import obj1, tools
import importlib

# %%
source = "UKB"

# %% [markdown]
# # Step 1. Load FR-reg

# %%
# load FR-reg
frreg_path = f"/data/jerrylee/pjt/BIGFAM.v.0.1/data/{source}/frreg/DOR"
fns = os.listdir(frreg_path)
len(fns)

# %% [markdown]
# # Step 2. FRLog-reg

# %%
frlog_path = f"/data/jerrylee/pjt/BIGFAM.v.0.1/data/{source}/FRLogreg"
gsw_path = f"/data/jerrylee/pjt/BIGFAM.v.0.1/data/{source}/obj1"

# %%
for ii, fn in enumerate(tqdm(fns)):
    pheno_name = fn.split(".")[0]
    # load FR-reg
    frreg_fn = f"{frreg_path}/{fn}"
    df_frreg = pd.read_csv(frreg_fn, delim_whitespace=True)
    
    # resampling FR-reg to compute CIs
    df_lmbds = obj1.resampleFrregCoefficients(df_frreg, n_resample=100)    
    
    # slope test(FRLog-reg)
    df_frlogreg = obj1.familialRelationshipLogRegression(df_lmbds)
    sig = obj1._slopeSig(df_frlogreg["slope"])
    
    # save slope test results
    savefn_slopetest = f"{frlog_path}/{pheno_name}.FRLOG"
    (tools.raw2long(df_frlogreg, params=["slope", "intercept"])
     .to_csv(savefn_slopetest, sep='\t', index=False))
    df_frlogreg.to_csv(savefn_slopetest + "_raw",  sep='\t', index=False)
    
    # prediction
    df_gsw = obj1.prediction(df_lmbds, sig, print_prog=True)
    # save prediction results 
    savefn_pred = f"{gsw_path}/{pheno_name}.GSW"
    (tools.raw2long(df_gsw, params=df_gsw.columns)
     .to_csv(savefn_pred, sep='\t', index=False))
    df_gsw.to_csv(savefn_pred + "_raw",  sep='\t', index=False)

# %%



