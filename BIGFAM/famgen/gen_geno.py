from dataclasses import dataclass, field
import numpy as np

@dataclass
class FamGenoSimul:
    """Generate haplotye matrices for family.
    Attributes:
        num_variant (int) : 
        prop_common_rare : (number of common variant) / (number of rare variant). 0 is only rare variant, -1 is only common variant

    Raises:
        Exception: _description_
        Exception: _description_

    Returns:
        _type_: _description_
    """
    num_variant : int
    prop_common_rare : float # number of common variant over rare variant

    # post init
    maf_lim_common : list = field(default_factory=lambda: [0.05, 0.95])
    maf_lim_rare : list = field(default_factory=lambda: [0.01, 0.05])
    num_common : int = field(init=False)
    num_rare : int = field(init=False)

    def __post_init__(self):
        prop_common_rare = self.prop_common_rare

        if prop_common_rare == 0: # rare only
            self.num_common = 0
            self.num_rare = self.num_variant
            
        elif prop_common_rare == -1: # common only 
            self.num_common = self.num_variant
            self.num_rare = 0
        else:
            self.num_common = int(self.num_variant / (1 + self.prop_common_rare))
            self.num_rare = int(self.num_variant - self.num_common)
        
        self.maf_common = np.random.uniform(min(self.maf_lim_common), 
                                            max(self.maf_lim_common), 
                                            self.num_common)
        self.maf_rare = np.random.uniform(min(self.maf_lim_rare), 
                                        max(self.maf_lim_rare), 
                                        self.num_rare)

    def __generate_haplotype(self, num_sample: int, maf: np.ndarray, chr_type: str = "diploid") -> np.ndarray:
        """Generate a haplotype matrix for a given sample size and minor allele frequency.

        Args:
            num_sample (int): The number of samples to generate.
            maf (np.ndarray, shape=(m, )): The minor allele frequency.
            chr_type (str, optional): The chromosome type. Can be either "diploid" or "haploid". Defaults to "diploid".

        Returns:
            np.ndarray: A `(num_sample, num_common, 2)` matrix representing the haplotype of each sample.
        """
        n = num_sample
        m = len(maf)
        
        hap1_common = np.random.binomial(n=1, p=maf, size=(n, m)).astype(int)
        hap2_common = np.zeros((n, m)).astype(int) if chr_type == "haploid" else np.random.binomial(n=1, p=maf, size=(n, m)).astype(int)
        return np.dstack([hap1_common, hap2_common])

    def __random_inherit(self, haps_parent):
        """Randomly inherited one of the two haplotype variants.

        Args:
            haps_parent (np.ndarray, shape=(num_sample, num_variant, 2)) : _description_
        """
        n, m, _ = haps_parent.shape
        random_recom_mask = np.random.binomial(n=1, p=0.5, size=(n, m))
        return haps_parent[:, :, 0] * random_recom_mask + haps_parent[:, :, 1] * (1 - random_recom_mask)

    
    def generate_parent_haplotype(self, num_sample: int, chr_type: str = "diploid"):
        """Generate haplotypes for a given number of samples and chromosome type.

        Args:
            num_sample (int): The number of samples to generate.
            chr_type (str, optional): The chromosome type. Can be either "diploid" or "haploid". Defaults to "diploid".

        Returns:
            tuple: A tuple of two `np.ndarray` of shape `(num_sample, num_common, 2)` representing the common and rare haplotypes for each sample.
        """
        haps_common = self.__generate_haplotype(num_sample, self.maf_common, chr_type)
        haps_rare = self.__generate_haplotype(num_sample, self.maf_rare, chr_type)

        return haps_common, haps_rare

    def make_offspring_haplotype(self, haps_mother: np.ndarray, haps_father: np.ndarray, off_type: str, chr_type: str = "autosome"):
        """
        Args:
            haps_mother (np.ndarray, shape=(num_sample, num_variant, 2)) : 
            haps_father (np.ndarray, shape=(num_sample, num_variant, 2)) : 
            off_type (str) : "son" or "daughter"
            chr_type (str, optional) : Can be either "autosome" or "chrX". Defaults to "autosome".

        Returns:
            numpy array of shape (num_sample, num_variant, 2)
        """
        assert haps_mother.shape == haps_father.shape, "Input arrays must have the same shape."
        num_sample, num_variant, _ = haps_mother.shape

        hap_from_mom = self.__random_inherit(haps_mother)

        if chr_type == "autosome":
            hap_from_dad = self.__random_inherit(haps_father)
        elif chr_type == "chrX":
            if off_type == "son":
                hap_from_dad = np.zeros(shape=(num_sample, num_variant))
            elif off_type == "daughter":
                assert (sum(sum(haps_father[:, :, 1])) == 0), "X chromosome of father is NOT haploid (contains genotype coded as 2)."
                hap_from_dad = haps_father[:, :, 0]
            else:
                raise ValueError("`off_type` should be 'son' or 'daughter'.")
        else:
            raise ValueError("`chr_type` should be 'autosome' or 'chrX'.")

        return np.dstack([hap_from_mom.astype(int), hap_from_dad.astype(int)])
