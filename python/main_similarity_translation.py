'''
Created on 28.01.2019

@author: Lisa
'''

import numpy as np
import pandas
import data as data_mod

import os


def main():
    '''
    Translate eQTL-based tissue similarity data from GTEx Nature paper to my tissues.
    
    @see: Aguet, F. et al. Genetic effects on gene expression across human tissues. NATURE 550, 204–213 (2017).
    '''
    
    # read Michael's data and GTEx tissue similarity
    similarity_dir = os.path.join(data_mod._base_path_windows,  "tissueSimilarityFromNaturePaper")
    tissue_similarity_gtex = pandas.read_csv(os.path.join(similarity_dir, "cis_clustered.txt"), sep="\t", index_col=0)
    np.fill_diagonal(tissue_similarity_gtex.values, 1)
    
    # read my mapping from Michael's tissues to GTEx tissues
    tissue_mapping = pandas.read_csv(os.path.join(similarity_dir, "tissue_mapping_mscherer_GTEx.csv"), sep=";", index_col=0)
    tissue_mapping["GTEx"] = tissue_mapping["GTEx"].apply(lambda x: x.split(",") if x != "??" else [])
    
    # translate from GTEx similarities to MScherer similarities
    tissue_similarity_ms = pandas.DataFrame(index=tissue_mapping.index.values, 
                                            columns=tissue_mapping.index.values, dtype=np.float64)
    mean_similarity_gtex = np.nanmean(tissue_similarity_gtex.values)
    for j in range(tissue_similarity_ms.shape[0]):
        for i in range(j):
            targets_i = tissue_mapping.loc[tissue_mapping.index[i], "GTEx"]
            targets_j = tissue_mapping.loc[tissue_mapping.index[j], "GTEx"]
            # special case: one tissue has no correspondence: take mean pairwise similarity of GTEx data
            if (len(targets_i) == 0) | (len(targets_j) == 0):
                tissue_similarity_ms.iloc[i,j] = mean_similarity_gtex
                tissue_similarity_ms.iloc[j,i] = mean_similarity_gtex
                continue
            # collect pairwise similarity of all mapping targets (including nans)
            sim_ij = tissue_similarity_gtex.loc[targets_i, targets_j]
            sim_ji = tissue_similarity_gtex.loc[targets_j, targets_i]
            # save on both halfs of similarity matrix for my data (easier to access)
            tissue_similarity_ms.iloc[i,j] = np.nanmean(np.concatenate((sim_ij.values.flatten(), sim_ji.values.flatten())))
            tissue_similarity_ms.iloc[j,i] = tissue_similarity_ms.iloc[i,j]
    np.fill_diagonal(tissue_similarity_ms.values, 1)
    
    # write translated mapping
    tissue_similarity_ms.to_csv(os.path.join(similarity_dir, "cis_translated.csv"), sep=";")


if __name__ == "__main__":
    main()
