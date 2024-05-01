import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from . import tools
# import .tools as tools #from BIGFAM import tools

def rename_id(df, before_after={}):
    
    return df.copy().rename(columns=before_after)
    
def std_col(vals):
    return (vals - np.mean(vals)) / np.std(vals)
    
def warning_conditions(df_frreg):
    warning_messages = []
    
    # negative FR-reg slope
    n_neg_slope = sum(df_frreg["slope"] < 0)
    if n_neg_slope:
        msg = "Some FR-reg slope is NEGATIVE"
        warning_messages += [msg]
    
    # for d1 in sorted(df_frreg["DOR"].unique()):
    #     for d2 in sorted(df_frreg["DOR"].unique()):
    #         slope1 = df_frreg.loc[df_frreg["DOR"] == d1, "slope"].values[0]
    #         slope2 = df_frreg.loc[df_frreg["DOR"] == d2, "slope"].values[0]
            
    #         if (d1 < d2) & (slope1 < slope2):
    #             msg = "Distance relatives has larger FR-reg slope"
    #             warning_messages += [msg]
        
    return warning_messages
  
def get_sextype(df):
    return (df[["volsex", "relsex"]]
            .apply(lambda x: "".join(map(str, sorted(x))), axis=1))

def remove_outliers(df, coln, verbose=False):
    less = -3 * np.std(df[coln])
    more = 3 * np.std(df[coln])
    outliers = (df[coln] < less) | (df[coln] > more)
    
    if verbose:
        print(sum(outliers), "removed from", len(df), flush=True)
        
    return df[~outliers].copy().reset_index(drop=True)
    
def merge_pheno_info(df_pheno, df_info):
    # validate input
    df_pheno = tools.check_columns(df_pheno.copy(), 
                                   column_types={
                                       "eid": int, 
                                       "pheno": float
                                       }
                                   )
    df_info = tools.check_columns(df_info.copy(),
                                  column_types={
                                       "DOR": int, 
                                       "volid": int, 
                                       "relid": int, 
                                       }
                                   ) 
    df_merge = pd.merge(
        pd.merge(df_info, 
                 rename_id(df_pheno, {"eid": "volid", "pheno": "volpheno"}), on="volid"),
        rename_id(df_pheno, {"eid": "relid", "pheno": "relpheno"}),
        on="relid")
    
    return df_merge


def familial_relationship_regression_DOR(df, std_pheno=True):
    # unique id set
    df = tools.remove_duplicate_relpair(df, ["volid", "relid"])
    
    # flip and concat volid and relid
    colns_to_flip = {"volid": "relid", 
                     "volage": "relage", 
                     "volsex": "relsex"}
    df = (tools.flip_and_concat(df.copy(), colns_to_flip)
          .dropna()
          .reset_index(drop=True))
    
    # regression by each DOR
    df_frreg = pd.DataFrame(columns=["DOR", "slope", "se", "p", "n"])
    
    for d in sorted(df["DOR"].unique()):
        df_dor = df[df["DOR"] == d].copy()
        
        if std_pheno:
            df_dor["volpheno"] = std_col(df_dor["volpheno"])
            df_dor["relpheno"] = std_col(df_dor["relpheno"])
        
        formula = "volpheno ~ 0 + relpheno"
        ll = smf.ols(formula, data=df_dor).fit()
        
        df_frreg.loc[len(df_frreg)] = [d, 
                                       ll.params[0], 
                                       ll.bse[0], 
                                       ll.pvalues[0], 
                                       ll.nobs]
        
    df_frreg = tools.check_columns(df_frreg.copy(), 
                                   column_types={
                                       "DOR": int, 
                                       "slope": float,
                                       "se": float,
                                       "p": float,
                                       "n": int,
                                       }
                                   )
    
    msgs = warning_conditions(df_frreg)
    
    return df_frreg, msgs


def familial_relationship_regression_REL(df, std_pheno=True, thred_pair=100):
    # remove no relationship information pairs
    df = df.dropna()
    
    # unique id set
    df = tools.remove_duplicate_relpair(df, ["volid", "relid"])
    
    # flip and concat volid and relid
    colns_to_flip = {"volid": "relid", 
                     "volage": "relage", 
                     "volsex": "relsex"}
    df = (tools.flip_and_concat(df.copy(), colns_to_flip)
          .dropna()
          .reset_index(drop=True))
    
    # regression by each REL
    df_frreg = pd.DataFrame(
        columns=["DOR", "relationship", "sex_type", "Erx", "slope", "se", "p", "n"]
        )
    
    for d in sorted(df["DOR"].unique()):
        df_dor = df[df["DOR"] == d].copy()
        
        for rel in sorted(df_dor["relationship"].unique()):
            df_rel = df_dor[df_dor["relationship"] == rel].copy()
            sex_type = (df_rel[["volsex", "relsex"]]
                        .apply(lambda x: "".join(map(str, sorted(x))), axis=1)
                        .unique()[0])
            Erx = df_rel["Erx"].unique()[0]
            
            if len(df_rel) < thred_pair: # < 5 relative pairs
                continue
            
            if std_pheno:
                df_rel["volpheno"] = std_col(df_rel["volpheno"])
                df_rel["relpheno"] = std_col(df_rel["relpheno"])
        
            formula = "volpheno ~ 0 + relpheno"
            ll = smf.ols(formula, data=df_rel).fit()
            
            df_frreg.loc[len(df_frreg)] = [d, rel, sex_type, Erx,
                                           ll.params[0], 
                                           ll.bse[0], 
                                           ll.pvalues[0], 
                                           ll.nobs]
        
    df_frreg = tools.check_columns(df_frreg.copy(), 
                                   column_types={
                                       "DOR": int, 
                                       "relationship": str,
                                       "slope": float,
                                       "se": float,
                                       "p": float,
                                       "n": int,
                                       }
                                   )
    
    msgs = warning_conditions(df_frreg)
    
    return df_frreg, msgs
    