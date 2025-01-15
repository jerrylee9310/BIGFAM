#! /usr/bin/env python
import math
import numpy as np
import numpy.linalg as la
import sympy
import pandas as pd
import os
from functools import reduce


def generate_geno(n, m, f, h2, out_prefix=False):
    """
    Generate sample genotype and phenotype with given heritability.
    (do not recomment to make too-big dataset cause it uses pandas.)

    [RETURN]
    g : genenotype
    y : phenotype
    """
    g = np.random.binomial(2, f, size=(n, m))
    g_df = pd.DataFrame()
    for idx in range(m):
        g_df["r" + str(idx + 1)] = g[:, idx]

    # save genotype matrix
    if out_prefix:
        g_df.to_csv(out_prefix + ".geno", sep='\t', index=False)

    g = g.astype(float)
    f = np.mean(g, axis=0) / 2

    # standardized genotype
    v = np.var(g, axis=0)
    z = (g - (2 * f)) / np.sqrt(v)

    # sample effects
    betas = np.random.normal(0, np.sqrt(h2 / m), size=m)

    # compute sample variances for betas and noise given h2
    ge = z.dot(betas)

    # ?? : what is s2g and s2e?
    s2g = np.var(ge) # theoretically, approx h2
    s2e = s2g * ((1.0 / h2) - 1)

    # create pheno
    e = np.random.normal(0, np.sqrt(s2e), size=n)
    y = ge + e

    # output phenotype mapping
    if out_prefix:
        with open("{}.phen".format(out_prefix), 'w') as phenfile:
            for idx, p in enumerate(y):
                fid = "{}".format(idx) # fID
                iid = "{}".format(idx) # iif
                phenfile.write("{} {} {}{}".format(fid, iid, p, os.linesep))

    return g, y

    
def make_grm(g, chr_type = "autosome", out_prefix=False):
    """
    Make GRM matrix

    g : genotype
    chr_type : "chrX" for X in males
    """
    n, m = g.shape

    # standarized genotype
    g = g.astype(float)

    if chr_type == "autosome":
        f = np.mean(g, axis=0) / 2

        v = np.var(g, axis=0)
        z = (g - (2 * f)) / np.sqrt(v)

        # compute GRM
        w = (1 / m) * z.dot(z.T)
    else:
        f = np.mean(g, axis=0)
        v = np.var(g, axis=0)
        z = (g - f) / np.sqrt(v)

        w = (1 / m) * z.dot(z.T)

    # output GRM files
    if out_prefix:
        pd.DataFrame(w).to_csv(out_prefix + ".grm", sep='\t', header=None, index=False)
        pd.DataFrame(m * np.ones((10, 10))).to_csv(out_prefix + ".grm.N", sep='\t', header=None, index=False) # number of sample used for calculating GRM (i,j) value
        pd.DataFrame(list(range(1, n+1))).to_csv(out_prefix + ".grm.id", sep='\t', header=None, index=False)

    return w



# LEGACY


def logdet(X):
    sign, ld = la.slogdet(X)
    if sign == 1:
        return ld
    else:
        return ld * -1


def dots(terms):
    return reduce(lambda x, y: x.dot(y), terms)


def get_terms(X, V):
    if X is not None:
        Vinv = la.inv(V)
        term = dots([X.T, Vinv, X])
        P = Vinv - dots([Vinv, X, la.inv(term), X.T, Vinv])
    else:
        Vinv = la.inv(V)
        P = Vinv
        term = None

    return (P, term)


# the log likelihood equation for the linear mixed model
def ll(y, X, P, V, term):
    if X is not None:
        return -0.5 * (logdet(V) + logdet(term) + dots([y.T, P, y]))
    else:
        return -0.5 * (logdet(V) + dots([y.T, P, y]))


def theory_se(A, P, Var):
    """ Method as described in ``Visscher et al 2014, Statistical Power to detect
        Genetic (Co)Variance of Complex Traits Using SNP Data in Unrelated Samples''.
        I think the formulated this be estimating the true population variance for pair-wise GRM...
        Not sure how well this holds up in real data.
    """
    n, n = A[0].shape
    return 10E5 / n**2


def delta_se(A, P, Var):
    """ Use the delta method to estimate the sample variance
    """
    r = len(A)
    S = np.zeros((r, r))
    N, N = A[0].shape

    # Calculate matrix for standard errors (same as AI matrix but w/out y)
    for i in range(r):
        for j in range(r):
            S[i, j] = np.trace(dots([P, A[i], P, A[j]]))

    S = 0.5 * S
    Sinv = la.inv(S)
    
    SE = np.zeros(r)
    xs = ["x{}".format(i) for i in range(r)]
    vars = sympy.Matrix(sympy.symbols(" ".join(xs)))
    sub=dict([(vars[i],Var[i]) for i in range(r)])
    
    for i in range(r):
        x = vars[i]
        exprs = [sympy.diff(vars[i] / sum(vars), x) for x in vars]
        grad = np.array([expr.evalf(subs=sub) for expr in exprs])
        SE[i] = dots([grad.T, Sinv / float(N), grad])

    return [np.sqrt(SE), Sinv]


def emREML(A, y, Var, X=None, calc_se=True, bounded=False, max_iter=100, verbose=False):
    """ Computes the ML estimate for variance components via the EM algorithm.
    A = GRM for variance component
    y = phenotype
    Var = array of initial estimates
    X = the design matrix for covariates
    if calc_se is true this returns
        var = variance estimates for each component
        se = the sample variance for each component's variance estimate
        sinv = the variance covariance matrix for estimates
    otherwise this returns
        var = variance estimates for each component
    """
    # N = float(len(y))
    N = len(y)

    # Add matrix of residuals to A
    A = [A, np.eye(N)]
    r = len(A)

    Var = np.var(y) * Var
    V = sum(A[i] * Var[i] for i in range(r))

    P, XtVinvX = get_terms(X, V)
    logL = ll(y, X, P, V, XtVinvX)
    if verbose:
        print('LogLike', 'V(G)', 'V(e)')

    l_dif = 10
    it = 0
    print("Iterate ML until convergence", flush=True)
    while it < max_iter and ( math.fabs(l_dif) >= 10E-4 or (math.fabs(l_dif) < 10E-2 and l_dif < 0) ):
        for i in range(r):
            vi2 = Var[i]
            vi4 = vi2**2
            Ai = A[i]
            Var[i] = (vi4 * dots([y.T, P, Ai, P, y]) + np.trace(vi2 * np.eye(N) - vi4 * P.dot(Ai)) ) / N
        V = sum(A[i] * Var[i] for i in range(r))

        P, XtVinvX = get_terms(X, V)
        new_logL = ll(y, X, P, V, XtVinvX)
        l_dif = new_logL - logL
        logL = new_logL
        it += 1

        if verbose:
            print(logL, Var[0], Var[1], flush=True)

    if not calc_se:
        final = Var
    else:
        SE, Sinv = delta_se(A, P, Var)
        final = [Var, SE, Sinv]

    return final


def aiREML(A, y, Var, X=None, calc_se=True, bounded=False, max_iter=100, verbose=False):
    """ Average Information method for computing the REML estimate of variance components.
    A = GRM for variance component
    y = phenotype
    Var = array of initial estimates
    X = the design matrix for covariates
    if calc_se is true this returns
        var = variance estimates for each component
        se = the sample variance for each component's variance estimate
        sinv = the variance covariance matrix for estimates
    otherwise this returns
        var = variance estimates for each component
    """
    # N = float(len(y))
    N = len(y)

    # Add matrix of residuals to A
    if type(A) == list:
        A = A + [np.eye(N)]
    else:
        A = [A, np.eye(N)]
    r = len(A)

    AI = np.zeros((r, r))
    s = np.zeros((r, 1))

    l_dif = 10
    it = 0

    Var = np.var(y) * Var

    # Perform a single iteration of EM-based REML to initiate parameters
    print("Perform a single iteration of EM-based REML to initiate parameters", flush=True)
    V = sum(A[i] * Var[i] for i in range(r))

    P, XtVinvX = get_terms(X, V)
    logL = ll(y, X, P, V, XtVinvX)

    for i in range(r):
        vi2 = Var[i]
        vi4 = vi2**2
        Ai = A[i]
        Var[i] = (vi4 * dots([y.T, P, Ai, P, y]) + np.trace(vi2 * np.eye(N) - vi4 * P.dot(Ai)) ) / N

    Var = isin_constraint(Var)

    V = sum(A[i] * Var[i] for i in range(r))

    P, XtVinvX = get_terms(X, V)
    logL = ll(y, X, P, V, XtVinvX)
    if verbose:
        print('LogLike', 'V(G)', 'V(e)', flush=True)

    # Iterate AI REML until convergence
    print("Iterate AI REML until convergence", flush=True)
    while it < max_iter and ( math.fabs(l_dif) >= 10E-4 or (math.fabs(l_dif) < 10E-2 and l_dif < 0) ):
        it = it + 1

        # Average information matrix
        for i in range(r):
            for j in range(r):
                # this is really just to slightly optimize...
                # Ai, Aj == I in the (r - 1) cases
                if i == (r - 1) and j == (r - 1):
                    AI[i, j] = dots([y.T, P, P, P, y]) 
                elif i == (r - 1):
                    AI[i, j] = dots([y.T, P, P, A[j], P, y]) 
                elif j == (r - 1):
                    AI[i, j] = dots([y, P, A[i], P, P, y])    
                else:
                    AI[i, j] = dots([y.T, P, A[i], P, A[j], P, y])  

        AI = 0.5 * AI

        # Vector of first derivatives of log likelihood function
        for i in range(r):
            # this is really just to slightly optimize...
            # Ai == I in the (r - 1) cases
            if i == (r - 1):
                s[i, 0] = np.trace(P) - dots([y.T, P, P, y]) 
            else:
                Ai = A[i]
                s[i, 0] = np.trace(P.dot(Ai)) - dots([y.T, P, Ai, P, y ])

        s = -0.5 * s

        # New variance components from AI and likelihood
        if l_dif > 1:
            # adjust for incomplete tagging according to GCTA-paper's coefficient
            Var = (Var + 0.316 * la.inv(AI).dot(s).T)[0]
        else:
            Var = (Var + la.inv(AI).dot(s).T)[0]

        Var = isin_constraint(Var)

        # Re-calculate V and P matrix
        V = sum(A[i] * Var[i] for i in range(r))

        # Likelihood of the MLM
        P, XtVinvX = get_terms(X, V)
        new_logL = ll(y, X, P, V, XtVinvX)
        l_dif = new_logL - logL
        logL = new_logL

        if verbose:
            print(logL, Var[0], Var[1], flush=True)

        if bounded:
            if min(Var/sum(Var)) < 0:
                break

    final = None
    if not calc_se:
        final = Var
    else:
        SE, Sinv = delta_se(A, P, Var)
        final = [Var, SE, Sinv]

    return final


def isin_constraint(Var, tol=1e-4):
    tmpVar = Var.copy()
    for i in range(len(tmpVar)):
        if tmpVar[i] < tol:
            tmpVar[i] = tol
            
    return tmpVar