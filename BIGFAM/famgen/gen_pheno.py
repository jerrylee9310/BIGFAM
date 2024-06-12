import pandas as pd
import numpy as np
from dataclasses import dataclass, field

def std_mtx(mtx):
    return (mtx - np.mean(mtx, axis=0)) / np.std(mtx, axis=0)

def set_var(var, rvs):
    return np.sqrt(var) * std_mtx(rvs)

def hap_to_geno(haps):
    return np.sum(haps, axis=2)

def make_pheno(hsq: float, haps: np.ndarray, beta: np.ndarray, dc_type: str = "None", n_cells=100):
    """_summary_

    Args:
        hsq (float) : variance of computed phenotype.
        haps (np.ndarray, shape=(n_sample, n_snp, 2)): _description_
        beta (np.ndarray, shape=(n_snp, )): _description_
        dc_type (str, optional): Either "None", "FDC", "NDC". if FDC, do XCI.
        n_cells (int, optional): _description_. Defaults to 100.
    """
    # check the given 
    if dc_type == "None":
        pheno = hap_to_geno(haps) @ beta
        new_hsq = hsq
    elif dc_type == "FDC":
        pheno = random_inactivation(haps, beta, n_cells)
        new_hsq = hsq / 2
    elif dc_type == "NDC":
        pheno = hap_to_geno(haps) @ beta
        new_hsq = 2 * hsq
    else:
        raise
    
    pheno = set_var(new_hsq, pheno)
    return pheno

def random_inactivation(haps_x, effect_size_x, n_cells=100):
    """
    Simulate random X-chromosome inactivation for female samples.
    
    Parameters:
    - haps_x (numpy array, shape=(n_sample, n_snp, 2)): haplotypes for the X-chromosome
    - effect_size_x (numpy array, shape=(n_snp,)): effect sizes for the X-chromosome
    - n_cells (int, optional): number of cells to simulate. Default is 100.
    
    Returns:
    - phenoByX (numpy array, shape=(n_sample,)): simulated phenotype values based on X-chromosome inactivation.
    """
    n_sample, n_snp, _ = haps_x.shape
    
    numCellactiveHap1 = np.random.binomial(n=1, p=0.5, size=(n_sample, n_cells)).sum(axis=1)
    numCellactiveHap2 = n_cells - numCellactiveHap1
    
    phenoByactiveHap = haps_x * effect_size_x[np.newaxis, :, np.newaxis]
    phenoByX = ((numCellactiveHap1[:, np.newaxis] * phenoByactiveHap[:, :, 0] + 
                 numCellactiveHap2[:, np.newaxis] * phenoByactiveHap[:, :, 1]) / n_cells).sum(axis=1)
    
    return phenoByX



