import pandas as pd
import numpy as np
from . import Xoptim
import statsmodels.formula.api as smf
from scipy.optimize import minimize
pd.set_option('mode.chained_assignment',  None) 

###

def _matchType(df_frreg):
    col_dicts = {"DOR": int,
                "relationship": str,
                "sex_type": str,
                "slope": float,
                "se": float,
                "Erx": float,
                "n": int}
    
    for coln, col_type in col_dicts.items():
        if not (coln in df_frreg.columns):
            continue
        df_frreg = df_frreg.astype({coln: col_type})
    return df_frreg

def _parsingCols(df_rel):
    return df_rel.T.to_dict()[df_rel.index[0]]
     
def _resamplingFRregCoefficients(df_lmbds, n_resample=100):#, n_block=10):
    """
    df_lmbds : summary of FR-reg
    """
    df_block = pd.DataFrame()

    for relationship in df_lmbds["relationship"].unique():
        df_rel = df_lmbds[df_lmbds["relationship"] == relationship].copy()
        vals = _parsingCols(df_rel) #dor, relationship, sex_type, Erx, slope, se = df_rel.iloc[0].values

        resampled_slopes = np.random.normal(vals["slope"], 
                                            vals["se"],
                                            n_resample) 
                                            # (n_block, n_resample // n_block))
        resampled_slopes = resampled_slopes.flatten()

        df_tmp = pd.DataFrame({
            "DOR": vals["DOR"],
            "relationship": relationship,
            "sex_type": vals["sex_type"],
            "Erx": vals["Erx"],
            "slope": resampled_slopes,
            "block": np.arange(n_resample) #np.repeat(np.arange(n_block), n_resample // n_block),
        })

        df_block = pd.concat([df_block, df_tmp], ignore_index=True)

    return df_block

def _regressOutMean(df_block, bin=["DOR"]):
    df_res = df_block.copy()

    def regress_out_mean(group):
        ll = smf.ols(formula="slope ~ 1", data=group).fit()
        group["eta"] = 2**group["DOR"].iloc[0] * ll.params["Intercept"]
        group["residual"] = ll.resid
        group["tl"] = group["Erx"] - group["Erx"].mean()
        return group

    df_res = df_res.groupby(bin).apply(regress_out_mean)

    return df_res

def _get_h2(df_frreg):
    """Get meta-h2"""
    
    def meta_h2(means, ses):
        """IVW"""
        ses = np.array(ses)
        means = np.array(means)
        meta_se = np.sqrt(1/np.sum(1/ses**2))
        meta_mean = np.sum(means/ses**2) * (meta_se**2)
        
        return meta_mean, meta_se
    
    means = []
    ses = []
    
    for d in sorted(df_frreg["DOR"].unique()):
        means_d = df_frreg.loc[df_frreg["DOR"] == d, "slope"].to_numpy()
        ses_d = df_frreg.loc[df_frreg["DOR"] == d, "se"].to_numpy()
        mean_d, se_d = meta_h2(2**d * means_d, 2**d * ses_d)
        means.append(mean_d)
        ses.append(se_d)
    
    return meta_h2(means, ses)
        


########

def estimateX(
    df_frreg, 
    n_resample=100,
    regout_bin=["DOR"],
    alpha_dicts={"type": "lambda", "weight":2},
    print_summary=True,
    ):
    """
    alpha_dicts : 
        - if "type" == "lambda", alpha is computed as lambda**"weight"
        - if "type" != "lambda", specify the alpha value in "weight"
    """
    # validate input
    df_frreg = _matchType(df_frreg)
    
    # parameters
    meta_lambda, _ = _get_h2(df_frreg)
    weight = float(alpha_dicts["weight"])
    df_lmbds = _resamplingFRregCoefficients(
        df_frreg,
        n_resample=n_resample
    )
    
    df_raw = pd.DataFrame(columns=["lambda", "alpha", "X"])
    
    for ib in range(n_resample):
        df_block = df_lmbds[df_lmbds["block"] == ib].copy()
        
        # compute FRresidual & eta
        df_block = _regressOutMean(df_block, bin=regout_bin)
        
        # L2 weight value
        if alpha_dicts["type"] == "lambda":
            alpha = float(1/meta_lambda)**weight
        else:
            alpha = alpha_dicts["weight"]
        
        if alpha < 0:
            print("L2 weight is negative...", flush=True)
            continue
        
        # estimate X
        MODEL = Xoptim.optToFindX(df_block, alpha)
        df_raw.loc[len(df_raw)] = [meta_lambda, alpha, MODEL.x[0]]

    if print_summary:
        print("""
              Prediction Result : 
              - Vx : {x:.3f}({x_lower:.3f},{x_upper:.3f})
              """.format(
              x = np.median(df_raw["X"]),
              x_lower = np.percentile(df_raw["X"], 2.5),
              x_upper = np.percentile(df_raw["X"], 97.5),
          ),
          flush=True
          )
    
    return df_raw

    
# def estimateXmXf(df_frreg, 
#                  n_resample=100,
#                  alpha_dicts={"type": "eta", "weight":-2},
#                  regout_bin=["DOR", "sex_type"]):
#     df_frreg = _matchType(df_frreg)
#     df_lmbds = _resamplingFRregCoefficients(df_frreg, n_resample=n_resample)
#     mean_eta, _ = _get_h2(df_frreg)
    
#     df_raw = pd.DataFrame(columns=["eta", "alpha", "Xmale", "Xfemale"])
#     for ib in range(n_resample):
#         df_block = df_lmbds[df_lmbds["block"] == ib].copy()
#         # compute FRresidual & eta
#         df_block = _regressOutMean(df_block, bin=regout_bin)
        
#         # L2 weight value
#         # mean_eta = df_block["eta"].mean()
#         if alpha_dicts["type"] == "eta":
#             alpha = float(1/mean_eta)**alpha_dicts["weight"]
#         else:
#             alpha = alpha_dicts["weight"]
        
#         if alpha < 0:
#             print("L2 weight is negative...", flush=True)
#             continue
        
#         # estimate X
#         for r0 in np.linspace(-1, 1, 11):
#             MODEL = _optToFindXmXf(df_block, alpha)
#             df_raw.loc[len(df_raw)] = [mean_eta, alpha, MODEL.x[0], MODEL.x[1]]

#     return df_raw

    
def estimateXmXfR(
    df_frreg, 
    n_resample=100,
    regout_bin=["DOR", "sex_type"],
    alpha_dicts={"type": "lambda", "weight": 2},
    print_summary=True,
    ):
    
    # validate input
    df_frreg = _matchType(df_frreg)
    
    # parameters
    meta_lambda, _ = _get_h2(df_frreg)
    weight = float(alpha_dicts["weight"])
    df_lmbds = _resamplingFRregCoefficients(
        df_frreg,
        n_resample=n_resample
    )
    
    df_raw = pd.DataFrame(columns=[
        "lambda", "alpha", "Xmale", "Xfemale", "r", 
    ])
    
    for ib in range(n_resample):
        tmp_block = pd.DataFrame(columns=[
            "Xmale", "Xfemale", "r", "func_val"
        ])
        df_block = df_lmbds[df_lmbds["block"] == ib].copy()
        
        # compute FRresidual & eta
        df_block = _regressOutMean(df_block, bin=regout_bin)
        
        # L2 weight value
        if alpha_dicts["type"] == "lambda":
            alpha = float(1/meta_lambda)**weight
        else:
            alpha = alpha_dicts["weight"]
        
        if alpha < 0:
            print("L2 weight is negative...", flush=True)
            continue
        
        # estimate X
        for r0 in np.linspace(-1, 1, 11):
            MODEL = Xoptim.optToFindXmXfR(df_block, alpha, r0)
            tmp_block.loc[len(tmp_block)] = [
                MODEL.x[0], MODEL.x[1], r0, MODEL.fun
                ]
        
        # min func value
        tmp_min = tmp_block[tmp_block["func_val"] == tmp_block["func_val"].min()]
        
        
        df_raw.loc[len(df_raw)] = [
            meta_lambda, alpha,
            tmp_min["Xmale"].values[0], 
            tmp_min["Xfemale"].values[0], 
            tmp_min["r"].values[0]
        ]

    if print_summary:
        print("""
              Prediction Result : 
              - Vx_male : {Xmale:.3f}({Xmale_lower:.3f},{Xmale_upper:.3f})
              - Vx_female : {Xfemale:.3f}({Xfemale_lower:.3f},{Xfemale_upper:.3f})
              - r : {r:.3f}({r_lower:.3f},{r_upper:.3f})
              """.format(
              Xmale = np.median(df_raw["Xmale"]),
              Xmale_lower = np.percentile(df_raw["Xmale"], 2.5),
              Xmale_upper = np.percentile(df_raw["Xmale"], 97.5),
              Xfemale = np.median(df_raw["Xfemale"]),
              Xfemale_lower = np.percentile(df_raw["Xfemale"], 2.5),
              Xfemale_upper = np.percentile(df_raw["Xfemale"], 97.5),
              r = np.median(df_raw["r"]),
              r_lower = np.percentile(df_raw["r"], 2.5),
              r_upper = np.percentile(df_raw["r"], 97.5),
          ),
          flush=True
          )
    
    return df_raw.reset_index(drop=True)
