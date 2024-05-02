import pandas as pd
import numpy as np
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
        


def _lossFuncX(x, df, alpha):
    # Fidelity term
    loss_fid = np.sum((df["residual"] - df["tl"] * x) ** 2)

    # L2 term
    loss_l2 = alpha * (x ** 2)

    return loss_fid + loss_l2

def _lossFuncXmXf(xs, df_alpha):
    x_male, x_female = xs
    df, alpha = df_alpha
        
    df_mm = df[df["sex_type"] == "MM"]
    df_mf = df[df["sex_type"] == "MF"]
    df_ff = df[df["sex_type"] == "FF"]
    
    # fidelity term
    loss_mm = np.sum((df_mm["residual"] - df_mm["tl"] * x_male)**2)
    loss_mf = np.sum((df_mf["residual"] - df_mf["tl"] * np.sqrt((x_male * x_female)))**2)
    loss_ff = np.sum((df_ff["residual"] - df_ff["tl"] * x_female)**2)
    
    # L2 term
    loss_l2 = alpha * (x_male**2 + x_female**2)
    
    return loss_mm + loss_mf + loss_ff + loss_l2

# def _lossFuncXmXfR(xs, df_alpha):
#     x_male, x_female = xs
#     df, alpha, r = df_alpha
    
#     df_mm = df[df["sex_type"] == "mm"]
#     df_mf = df[df["sex_type"] == "mf"]
#     df_ff = df[df["sex_type"] == "ff"]
    
#     # fidelity term
#     loss_mm = np.sum((df_mm["residual"] - df_mm["tl"] * x_male)**2)
#     loss_mf = np.sum((df_mf["residual"] - df_mf["tl"] * r * np.sqrt((x_male * x_female)))**2)
#     loss_ff = np.sum((df_ff["residual"] - df_ff["tl"] * x_female)**2)
    
#     # L2 term
#     loss_l2 = alpha * (x_male**2 + x_female**2)
    
#     return loss_mm + loss_mf + loss_ff + loss_l2


def _optToFindX(df_block, alpha, lower_lim=-1, upper_lim=1):
    # optimization
    x0 = [0.01]
    bnds = [(lower_lim, upper_lim)] # [(1e-6, 1)]
    
    model = minimize(
        fun=_lossFuncX,
        x0=x0,
        args=(df_block, alpha),
        bounds=bnds,
        tol=1e-4
    )

    return model

def _optToFindXmXf(df_block, alpha, lower_lim=1e-6, upper_lim=1):
    # optimization
    x0 = [0.01, 0.01] # Xmale, Xfemale
    bnds = [(lower_lim, upper_lim), (lower_lim, upper_lim)] # [(1e-6, 1)]
    
    MODEL = minimize(
        fun=_lossFuncXmXf,
        x0=x0,
        args=[df_block, alpha],
        bounds=bnds,
        tol=1e-4
    )

    return MODEL

# def _optToFindXmXfR(df_block, alpha, r0, lower_lim=1e-6, upper_lim=1):
#     # optimization
#     x0 = [0.01, 0.01] # Xmale, Xfemale
#     bnds = [(lower_lim, upper_lim), (lower_lim, upper_lim)] # [(1e-6, 1)]
    
#     MODEL = minimize(
#         fun=_lossFuncXmXfR,
#         x0=x0,
#         args=[df_block, alpha, r0],
#         bounds=bnds,
#         tol=1e-6
#     )
#     return MODEL

########

def estimateX(df_frreg, 
              n_resample=100,
              alpha_dicts={"type": "eta", "weight":-2},
              regout_bin=["DOR"]):
    """
    alpha_dicts : 
        - if "type" == "eta", alpha is computed as eta**"weight"
        - if "type" != "eta", specify the alpha value in "weight"

    """
    df_frreg = _matchType(df_frreg)
    mean_eta, _ = _get_h2(df_frreg)
    
    df_lmbds = _resamplingFRregCoefficients(df_frreg, n_resample=n_resample)
    
    df_raw = pd.DataFrame(columns=["eta", "alpha", "X"])
    for ib in range(n_resample):
        df_block = df_lmbds[df_lmbds["block"] == ib].copy()
        # compute FRresidual & eta
        df_block = _regressOutMean(df_block, bin=regout_bin)
        
        # L2 weight value
        if alpha_dicts["type"] == "eta":
            alpha = float(1/mean_eta)**float(alpha_dicts["weight"])
        else:
            alpha = alpha_dicts["weight"]
        
        if alpha < 0:
            print("L2 weight is negative...", flush=True)
            continue
        
        # estimate X
        MODEL = _optToFindX(df_block, alpha)
        df_raw.loc[len(df_raw)] = [mean_eta, alpha, MODEL.x[0]]

    return df_raw

    
def estimateXmXf(df_frreg, 
                 n_resample=100,
                 alpha_dicts={"type": "eta", "weight":-2},
                 regout_bin=["DOR", "sex_type"]):
    df_frreg = _matchType(df_frreg)
    df_lmbds = _resamplingFRregCoefficients(df_frreg, n_resample=n_resample)
    
    df_raw = pd.DataFrame(columns=["eta", "alpha", "Xmale", "Xfemale"])
    for ib in range(n_resample):
        df_block = df_lmbds[df_lmbds["block"] == ib].copy()
        # compute FRresidual & eta
        df_block = _regressOutMean(df_block, bin=regout_bin)
        
        # L2 weight value
        mean_eta = df_block["eta"].mean()
        if alpha_dicts["type"] == "eta":
            alpha = mean_eta**alpha_dicts["weight"]
        else:
            alpha = alpha_dicts["weight"]
        
        if alpha < 0:
            print("L2 weight is negative...", flush=True)
            continue
        
        # estimate X
        MODEL = _optToFindXmXf(df_block, alpha)
        df_raw.loc[len(df_raw)] = [mean_eta, alpha, MODEL.x[0], MODEL.x[1]]

    return df_raw

    
# def estimateXmXfR(df_frreg, 
#                  n_resample=100,
#                  alpha_dicts={"type": "eta", "weight":-2},
#                  regout_bin=["DOR", "sex_type"]):
#     df_frreg = _matchType(df_frreg)
#     df_lmbds = _resamplingFRregCoefficients(df_frreg, n_resample=n_resample)
    
#     df_raw = pd.DataFrame(columns=["block_idx", "eta", "alpha", "Xmale", "Xfemale", "r", "func_val"])
    
#     for ib in range(n_resample):
#         df_block = df_lmbds[df_lmbds["block"] == ib].copy()
#         # compute FRresidual & eta
#         df_block = _regressOutMean(df_block, bin=regout_bin)
        
#         # L2 weight value
#         mean_eta = df_block["eta"].mean()
#         if alpha_dicts["type"] == "eta":
#             alpha = mean_eta**alpha_dicts["weight"]
#         else:
#             alpha = alpha_dicts["weight"]
        
#         if alpha < 0:
#             print("L2 weight is negative...", flush=True)
#             continue
        
#         # estimate X
#         for r0 in np.linspace(-1, 1, 11):
#             MODEL = _optToFindXmXfR(df_block, alpha, r0)
#             df_raw.loc[len(df_raw)] = [ib, mean_eta, alpha, MODEL.x[0], MODEL.x[1], r0, MODEL.fun]

#     return df_raw
