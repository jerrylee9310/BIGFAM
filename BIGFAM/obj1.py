import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.optimize import minimize

###

def _checkNegativeCoefficients(df_lmbds):
    neg_flag = False
    
    if sum(df_lmbds.groupby("DOR").mean()["slope"] < 0):
        neg_flag = True
        print("More than one FR-reg coefficient can be < 0", 
              flush=True)
        
    return neg_flag

def _generateCVIndex(n, m):
    """
    n : range of integers
    m : number of repeats
    """
    # Create a list with integers from 0 to 9 repeated 10 times
    original_list = list(range(n))
    repeated_list = original_list * m
    
    # Shuffle the list to get a random order
    np.random.shuffle(repeated_list)
    
    return repeated_list

def _labelResampledCoefficients(df_lmbds, n_block=None):
    if n_block is None:
        num_of_lmbd_in_each_dor = df_lmbds.groupby("DOR").size().to_list()
        if len(set(num_of_lmbd_in_each_dor)) == 1:
            n_block = num_of_lmbd_in_each_dor[0]
            
    # indexing for CV
    for d in df_lmbds["DOR"].unique():
        is_dor = df_lmbds["DOR"] == d
        
        # randomly group 100 lmbds
        n_lmbds_in_block = int(df_lmbds[is_dor].shape[0] / n_block)
        df_lmbds.loc[is_dor, "block"] = _generateCVIndex(n_block, n_lmbds_in_block)
    
    return df_lmbds

def _logRegression(df_lmbds):
    slopes = []
    intercepts = []
    indices = []
    for idx in df_lmbds["block"].unique():
        df_tmp = df_lmbds[df_lmbds["block"] == idx]
        ll = smf.ols("log_slope ~ 1 + neg_dor", data=df_tmp).fit()
        slopes += [ll.params["neg_dor"]]
        intercepts += [ll.params["Intercept"]]
        indices += [idx]
    
    df_frlog = pd.DataFrame({"block": indices,
                             "slope": slopes,
                             "intercept": intercepts},
                            )
    return df_frlog

def _lossFunc(xs, df_lmbds_w0):
    G, S1 = xs
    df_lmbds, w0 = df_lmbds_w0
    
    df_lmbds["slope"] = df_lmbds["slope"].astype(float)
    df_lmbds["log_slope"] = np.log2(df_lmbds["slope"])
    df_lmbds["slope_pred"] = (1/2)**df_lmbds["DOR"] * G\
                           + w0**(df_lmbds["DOR"] - 1) * S1
    df_lmbds["log_slope_pred"] = np.log2(df_lmbds["slope_pred"])
    df_lmbds["log_slope_pred"] = df_lmbds["log_slope_pred"].astype(float)
    return sum((df_lmbds["log_slope"] - df_lmbds["log_slope_pred"])**2)

def _gridSearch(df_train, df_test, w_range):
    df_pred = pd.DataFrame(columns=["V(g)", "V(s)", "w", "mse_train", "mse_test"])
    df_train = df_train.copy()
    df_test = df_test.copy()
    
    for w in w_range:
        x0 = [0.5, 0.1] # initial V_g, V_s
        bnds = ((1e-6, 1), (1e-6, 1))
        
        # in train data 
        MODEL = minimize(
            fun = _lossFunc,
            x0 = x0,
            args = [df_train, w],
            bounds = bnds,
            tol=1e-4
        )
        
        G_est, S1_est = MODEL.x
        
        # in test data
        df_test["log_slope"] = np.log2(df_test["slope"])
        df_test["slope_pred"] = (1/2)**df_test["DOR"] * G_est\
                            + (w)**(df_test["DOR"] - 1) * S1_est
        df_test["log_slope_pred"] = np.log2(df_test["slope_pred"])
        mse = sum((df_test["log_slope"] - df_test["log_slope_pred"])**2)
        
        df_pred.loc[len(df_pred)] = [G_est, S1_est, w, MODEL.fun, mse]

    return df_pred

def _setInitialRange(frlogreg_sig, step_size):
    if frlogreg_sig == "None":
        w0_range = np.arange(0.4, 0.6+step_size, step_size)
    elif frlogreg_sig == "Low":
        w0_range = np.arange(0.55, 0.95+step_size, step_size)
    elif frlogreg_sig == "High":
        w0_range = np.arange(0.01, 0.45+step_size, step_size)
    
    return w0_range

def _getMin(df_pred):
    min_row = df_pred.loc[df_pred["mse_test"].idxmin()]
    
    return pd.DataFrame(min_row).T
    
def _slopeSig(slopes):
    sig = "None"
    
    lower = np.percentile(slopes, 2.5)
    upper = np.percentile(slopes, 97.5)
    
    if lower > 1:
        sig = "High"
    if upper < 1:
        sig = "Low"
        
    return sig
    
####

def resampleFrregCoefficients(df_frreg, n_resample=100):
    df_resample = pd.DataFrame(columns=["DOR", "idx", "slope"])
    
    for dor in df_frreg["DOR"].unique():
        slope, se = df_frreg.loc[df_frreg["DOR"] == dor, ["slope", "se_slope"]].values[0]
            
        df_tmp = pd.DataFrame({"DOR": dor,
                               "idx": range(n_resample),
                               "slope": np.abs(np.random.normal(slope, se, size=n_resample))})
        
        df_resample = pd.concat([df_resample, df_tmp], axis=0).reset_index(drop=True)
    
    neg_flag = _checkNegativeCoefficients(df_resample)
    
    if neg_flag:
        print("FR-reg coefficient can be negative..", flush=True)
    
    return df_resample
    
def familialRelationshipLogRegression(df_lmbds):
    df_lmbds = df_lmbds.copy()
    df_frlogreg = pd.DataFrame(columns=["slope", "lower_slope", "upper_slope",
                                        "intercept", "lower_intercept", "upper_intercept"])
    df_lmbds["neg_dor"] = -df_lmbds["DOR"]
    df_lmbds["neg_dor"] = df_lmbds["neg_dor"].astype(float)
    df_lmbds["log_slope"] = np.log2(df_lmbds["slope"])
    
    # random indexing
    df_block = _labelResampledCoefficients(df_lmbds)
    df_frlog = _logRegression(df_block)
    
    df_frlogreg.loc[len(df_frlogreg)] = [np.median(df_frlog["slope"]), np.quantile(df_frlog["slope"], 0.025), np.quantile(df_frlog["slope"], 0.975),
                                         np.median(df_frlog["intercept"]), np.quantile(df_frlog["intercept"], 0.025), np.quantile(df_frlog["intercept"], 0.975)]
    # return df_frlogreg
    return df_frlog
     
def prediction(df_lmbds, frlogreg_sig, 
                   n_repeat_cv=10, n_block=10, 
                   step_size=0.01, 
                   print_prog=False):
    df_res = pd.DataFrame()
    df_lmbds = df_lmbds.astype({"DOR": int,
                                "idx":int,
                                "slope":float})
    
    # set grid search range 
    w0_range = _setInitialRange(frlogreg_sig, step_size)
    
    for i_rcv in range(n_repeat_cv):
        if print_prog: print(f"{i_rcv + 1}..", end="", flush=True)
        
        # randomly block the FR-reg coefficients
        df_frreg_with_block = _labelResampledCoefficients(df_lmbds, n_block=n_block)
        
        # cross validataion
        for i_b in range(n_block):
            df_train = df_frreg_with_block[df_frreg_with_block["block"] != i_b].copy()
            df_test = df_frreg_with_block[df_frreg_with_block["block"] == i_b].copy()
            
            # grid search
            df_preds = _gridSearch(df_train, df_test, w0_range)
            df_best_pred = _getMin(df_preds)
            
            df_res = pd.concat([df_res, df_best_pred], axis=0).reset_index(drop=True)
            
    return df_res
            
    