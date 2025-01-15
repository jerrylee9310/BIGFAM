import pandas as pd
import numpy as np
from scipy.optimize import minimize

def _lossFuncX(x, df, alpha):
    # Fidelity term
    loss_fid = np.sum((df["residual"] - df["tl"] * x) ** 2)

    # L2 term
    loss_l2 = alpha * (x ** 2)

    return loss_fid + loss_l2

# def _lossFuncXmXf(xs, df_alpha):
#     x_male, x_female = xs
#     df, alpha = df_alpha
        
#     df_mm = df[df["sex_type"] == "MM"]
#     df_mf = df[(df["sex_type"] == "FM") | (df["sex_type"] == "MF")]
#     df_ff = df[df["sex_type"] == "FF"]
    
#     # fidelity term
#     loss_mm = np.sum((df_mm["residual"] - df_mm["tl"] * x_male)**2)
#     loss_mf = np.sum((df_mf["residual"] - df_mf["tl"] * np.sqrt((x_male * x_female)))**2)
#     loss_ff = np.sum((df_ff["residual"] - df_ff["tl"] * x_female)**2)
    
#     # L2 term
#     loss_l2 = alpha * (x_male**2 + x_female**2)
    
#     return loss_mm + loss_mf + loss_ff + loss_l2

def _lossFuncXmXfR(xs, df_alpha):
    x_male, x_female = xs
    df, alpha, r = df_alpha
    
    is_malepair = (df["sex_type"] == "MM")
    is_femalepair = (df["sex_type"] == "FF")
    
    df_mm = df[is_malepair]
    df_mf = df[~(is_malepair | is_femalepair)]
    df_ff = df[is_femalepair]
    
    # fidelity term
    loss_mm = np.sum((df_mm["residual"] - df_mm["tl"] * x_male)**2)
    loss_mf = np.sum((df_mf["residual"] - df_mf["tl"] * r * np.sqrt((x_male * x_female)))**2)
    loss_ff = np.sum((df_ff["residual"] - df_ff["tl"] * x_female)**2)
    
    # L2 term
    loss_l2 = alpha * (x_male**2 + x_female**2)
    
    return loss_mm + loss_mf + loss_ff + loss_l2


def optToFindX(df_block, alpha, lower_lim=-1, upper_lim=1):
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

# def optToFindXmXf(df_block, alpha, lower_lim=1e-6, upper_lim=1):
#     # optimization
#     x0 = [0.01, 0.01] # Xmale, Xfemale
#     bnds = [(lower_lim, upper_lim), (lower_lim, upper_lim)] # [(1e-6, 1)]
    
#     MODEL = minimize(
#         fun=_lossFuncXmXf,
#         x0=x0,
#         args=[df_block, alpha],
#         bounds=bnds,
#         tol=1e-4
#     )

#     return MODEL

def optToFindXmXfR(df_block, alpha, r0, lower_lim=1e-6, upper_lim=1):
    # optimization
    x0 = [0.01, 0.01] # Xmale, Xfemale
    bnds = [(lower_lim, upper_lim), (lower_lim, upper_lim)] # [(1e-6, 1)]
    
    MODEL = minimize(
        fun=_lossFuncXmXfR,
        x0=x0,
        args=[df_block, alpha, r0],
        bounds=bnds,
        tol=1e-6
    )
    return MODEL
