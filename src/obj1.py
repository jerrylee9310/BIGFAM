import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

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

# def _gridSearch(df_train, df_test, w_range):
#     df_pred = pd.DataFrame(columns=["V(g)", "V(s)", "w", "mse_train", "mse_test"])
#     df_train = df_train.copy()
#     df_test = df_test.copy()
    
#     for w in w_range:
#         x0 = [0.5, 0.1] # initial V_g, V_s
#         bnds = ((1e-6, 1), (1e-6, 1))
        
#         # in train data 
#         MODEL = minimize(
#             fun = _lossFunc,
#             x0 = x0,
#             args = [df_train, w],
#             bounds = bnds,
#             tol=1e-4
#         )
        
#         G_est, S1_est = MODEL.x
        
#         # in test data
#         df_test["log_slope"] = np.log2(df_test["slope"])
#         df_test["slope_pred"] = (1/2)**df_test["DOR"] * G_est\
#                             + (w)**(df_test["DOR"] - 1) * S1_est
#         df_test["log_slope_pred"] = np.log2(df_test["slope_pred"])
#         mse = sum((df_test["log_slope"] - df_test["log_slope_pred"])**2)
        
#         df_pred.loc[len(df_pred)] = [G_est, S1_est, w, MODEL.fun, mse]

#     return df_pred

def _single_grid_search(params):
    """단일 w 값에 대한 grid search를 수행하는 함수
    
    Args:
        params (tuple): (w, df_train, df_test) 형태의 튜플
        
    Returns:
        list: [G_est, S1_est, w, train_mse, test_mse] 형태의 결과
    """
    w, df_train, df_test = params
    x0 = [0.5, 0.1]  # initial V_g, V_s
    bnds = ((1e-6, 1), (1e-6, 1))
    
    # Train data에서 모델 학습
    MODEL = minimize(
        fun = _lossFunc,
        x0 = x0,
        args = [df_train, w],
        bounds = bnds,
        tol=1e-4
    )
    
    G_est, S1_est = MODEL.x
    
    # Test data에서 성능 평가
    df_test = df_test.copy()
    df_test["log_slope"] = np.log2(df_test["slope"])
    df_test["slope_pred"] = (1/2)**df_test["DOR"] * G_est + (w)**(df_test["DOR"] - 1) * S1_est
    df_test["log_slope_pred"] = np.log2(df_test["slope_pred"])
    test_mse = sum((df_test["log_slope"] - df_test["log_slope_pred"])**2)
    
    return [G_est, S1_est, w, MODEL.fun, test_mse]

def _gridSearch(df_train, df_test, w_range, max_workers=None):
    """병렬화된 grid search 함수
    
    Args:
        df_train (DataFrame): 학습 데이터
        df_test (DataFrame): 테스트 데이터
        w_range (array-like): 탐색할 w 값들
        max_workers (int, optional): 사용할 프로세스 수. None이면 CPU 코어 수를 사용
        
    Returns:
        DataFrame: Grid search 결과
    """
    df_pred = pd.DataFrame(columns=["V(g)", "V(s)", "w", "mse_train", "mse_test"])
    
    # 각 w 값에 대한 파라미터 준비
    params = [(w, df_train, df_test) for w in w_range]
    
    # 병렬 처리 실행
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_single_grid_search, params))
    
    # 결과를 DataFrame으로 변환
    df_pred = pd.DataFrame(
        results,
        columns=["V(g)", "V(s)", "w", "mse_train", "mse_test"]
    )
    
    return df_pred


def _setInitialRange(slope_lower_upper, step_size):
    lower, upper = slope_lower_upper
    
    w0_range = np.arange(0.4, 0.6+step_size, step_size)
    if lower > 1:
        w0_range = np.arange(0.01, 0.45+step_size, step_size)
    if upper < 1:
        w0_range = np.arange(0.55, 0.95+step_size, step_size)
    
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
        slope, se = df_frreg.loc[df_frreg["DOR"] == dor, ["slope", "se"]].values[0]
            
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
     
def prediction(
    df_lmbds: pd.DataFrame, 
    slope_lower_upper: list, 
    n_repeat_cv: int = 10, 
    n_block: int = 10, 
    step_size: float = 0.01, 
    print_prog: bool = False
    ):
    df_res = pd.DataFrame()
    df_lmbds = df_lmbds.astype({"DOR": int,
                                "idx":int,
                                "slope":float})
    
    # set grid search range 
    w0_range = _setInitialRange(slope_lower_upper, step_size)
    
    for i_rcv in range(n_repeat_cv):
        if print_prog: print(f"{i_rcv + 1}..", end="", flush=True)
        
        # randomly block the FR-reg coefficients
        df_frreg_with_block = _labelResampledCoefficients(
            df_lmbds, 
            n_block=n_block
        )
        
        # cross validataion
        for i_b in range(n_block):
            df_train = df_frreg_with_block[df_frreg_with_block["block"] != i_b].copy()
            df_test = df_frreg_with_block[df_frreg_with_block["block"] == i_b].copy()
            
            # grid search
            df_preds = _gridSearch(
                df_train, df_test, w0_range, 
                max_workers=cpu_count()//2
            )
            # df_preds = _gridSearch(df_train, df_test, w0_range)
            df_best_pred = _getMin(df_preds)
            
            df_res = pd.concat([df_res, df_best_pred], axis=0).reset_index(drop=True)
            
    return df_res
            
# def obj1(
#     df_frreg,
#     n_resample=100,
#     n_repeat_cv=10, 
#     n_block=10, 
#     step_size=0.01, 
#     print_prog=True,
#     ):
#     # resampling FR-reg to compute CIs
#     df_lmbds = resampleFrregCoefficients(
#         df_frreg, 
#         n_resample=n_resample
#     )    
    
#     # slope test(FRLog-reg)
#     df_frlogreg = familialRelationshipLogRegression(df_lmbds)
#     sig = _slopeSig(df_frlogreg["slope"])
    
#     print("""
#           Slope Test Result : {sig} (slope is {median:.3f}({lower:.3f}, {upper:.3f}))
#           """.format(
#               sig = sig,
#               median = np.median(df_frlogreg["slope"]),
#               lower = np.percentile(df_frlogreg["slope"], 2.5),
#               upper = np.percentile(df_frlogreg["slope"], 97.5),
#           ),
#           flush=True
#           )
    
#     print("""
#           Predict Variance Components...
#           """,
#           flush=True
#           )
    
#     # prediction
#     df_gsw = prediction(
#         df_lmbds, 
#         sig, 
#         n_repeat_cv=n_repeat_cv, 
#         n_block=n_block, 
#         step_size=step_size, 
#         print_prog=print_prog
#         )
    
#     print("""
#           Prediction Result : 
#           - Vg : {g:.3f}({g_lower:.3f},{g_upper:.3f})
#           - Vs : {s:.3f}({s_lower:.3f},{s_upper:.3f})
#           - ws : {w:.3f}({w_lower:.3f},{w_upper:.3f})
#           """.format(
#               g = np.median(df_gsw["V(g)"]),
#               g_lower = np.percentile(df_gsw["V(g)"], 2.5),
#               g_upper = np.percentile(df_gsw["V(g)"], 97.5),
#               s = np.median(df_gsw["V(s)"]),
#               s_lower = np.percentile(df_gsw["V(s)"], 2.5),
#               s_upper = np.percentile(df_gsw["V(s)"], 97.5),
#               w = 1 / np.median(df_gsw["w"]),
#               w_lower = 1 / np.percentile(df_gsw["w"], 2.5),
#               w_upper = 1 / np.percentile(df_gsw["w"], 97.5),
#           ),
#           flush=True
#           )
    
#     return df_frlogreg, df_gsw