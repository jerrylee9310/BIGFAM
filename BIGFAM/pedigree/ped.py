# GOAL : 
# infer familialial relaitonship 
# with genetic correlation on X

import numpy as np
import pandas as pd
from itertools import combinations
import copy
from .. import tools
from . import dfs
# from BIGFAM import tools
# from BIGFAM.pedigree import dfs
import importlib
importlib.reload(dfs)

relationships = ['SB', 'PC', 'AV', 'GP', '1C', 'HSB', 'HAV', 'GG']

dict_rel_rx = {
    "son_father": ["SF", 0],
    "son_mother": ["SM", 1/np.sqrt(2)],
    "daughter_father": ["DF", 1/np.sqrt(2)],
    "daughter_mother": ["DM", 1/2],
    "son_brother": ["SB", 1/2],
    "son_sister": ["SS+DB", 1/np.sqrt(8)],
    "daughter_sister": ["DS", 3/4],
}

def shortest_list(list_of_lists):
    if not list_of_lists:
        return None
    shortest = list_of_lists[0]
    for item in list_of_lists:
        if len(item) < len(shortest):
            shortest = item
    return shortest

def find_parents_with_sibling(df_ped: pd.DataFrame):
    return (df_ped.groupby(['father', 'mother'])
            .size()
            .to_frame(name='count')
            .query('count > 1')
            .index.to_list())
    
def find_sibling(df_parent: pd.DataFrame, fid: int, mid: int):
    """
    Finds sibling IDs for a given individual based on parents.

    This function searches the provided DataFrame (df_parent) for offspring 
    records where both "father" and "mother" IDs match the given IDs (fid and mid). 
    It then returns a list of sibling IDs ("offspring") for the individual.

    Args:
        df_parent (pd.DataFrame): DataFrame containing parent-offspring relationships.
        fid (int): Father ID of the individual.
        mid (int): Mother ID of the individual.

    Returns:
        list: List of sibling IDs for the given individual.
    """

    # Filter DataFrame for offspring with matching parents
    df_offspring = df_parent[(df_parent["father"] == fid) & (df_parent["mother"] == mid)]

    # Extract and return sibling IDs
    sibling_ids = df_offspring["offspring"].tolist()  # Use tolist() for a list
    return sibling_ids

def validate_input(
    df_ped: pd.DataFrame,  
    ids: list,             
    relationship: str      
):
    """
    Validates input data for pedigree analysis.

    This function performs the following checks on the input data:

    - Ensures the provided DataFrame (df_ped) has the following columns with specific data types:
        - "offspring": Integer
        - "sex": String ("M" or "F")
        - "father": Integer
        - "mother": Integer
    - Verifies that the 'ids' list contains exactly two integer values.
    - Confirms that the 'relationship' string is a valid relationship type (defined elsewhere).

    Args:
        df_ped (pd.DataFrame): Pandas DataFrame containing pedigree data.
        ids (list): List of two integer IDs for individuals.
        relationship (str): String representing the relationship between the IDs.

    Returns:
        pd.DataFrame: The validated and potentially modified DataFrame (df_ped).
        list: The validated list of IDs.
        str: The validated relationship string.

    Raises:
        AssertionError: If any of the validation checks fail.
    """

    # Validate DataFrame columns and data types
    df_ped = (tools.check_columns(
        df=df_ped.copy(),  # Copy the DataFrame to avoid modifying the original
        column_types={
            "offspring": int,
            "sex": str,
            "father": int,
            "mother": int
        },
        column_components={"sex": ["M", "F"]})
        .reset_index(drop=True))  # Reset index after potential modifications

    # Assert conditions for valid input
    assert len(ids) == 2, "List 'ids' must contain exactly two IDs."
    assert relationship in relationships, f"Invalid relationship type: '{relationship}'"

    return df_ped, ids, relationship

def sort_id_by_age(df_pair: pd.DataFrame):
    """
    Sorts a DataFrame containing paired IDs based on age.

    This function assumes the DataFrame (df_pair) has the following columns:

      - "eid": Integer ID of the first individual.
      - "eid_rel": Integer ID of the related individual.
      - "age": Integer age of the first individual.
      - "age_rel": Integer age of the related individual.

    The function sorts the DataFrame such that the younger individual's ID 
    appears in the "eid" column and the older individual's ID appears in 
    "eid_rel" column.

    Args:
        df_pair (pd.DataFrame): DataFrame containing paired IDs and ages.

    Returns:
        pd.DataFrame: The sorted DataFrame with younger IDs in the "eid" column.
    """

    # Validate DataFrame columns (already assumed in your code)
    df_pair = tools.check_columns(
        df=df_pair.copy(),
        column_types={
            "eid": int,
            "eid_rel": int,
            "age": int,
            "age_rel": int
        }
    )

    # Use vectorized approach for efficiency
    df_sorted = df_pair[df_pair["age"] < df_pair["age_rel"]]  # Select younger rows
    df_sorted = pd.concat([df_sorted, df_pair[df_pair["age"] >= df_pair["age_rel"]]])  # Add older rows
    
    return df_sorted.reset_index(drop=True)

def make_direct_edges(df_ped: pd.DataFrame):
    df_edges = pd.DataFrame(columns=["id1", "id2", "relationship"])
    
    # make full sibling edges
    parents = find_parents_with_sibling(df_ped)
    sibs = []
    for father_id, mother_id in parents:
        sibling_ids = find_sibling(df_ped, father_id, mother_id)
        sibs.append(sibling_ids)
    
    # add offspring-parent
    for _, row in df_ped.iterrows():
        id_off, sex_off = row[["offspring", "sex"]]
        
        for p in ["father", "mother"]:
            p_id = row[p]
            if sex_off == "M":
                df_edges.loc[len(df_edges)] = [id_off, p_id, f"son_{p}"]
            elif sex_off == "F":
                df_edges.loc[len(df_edges)] = [id_off, p_id, f"daughter_{p}"]
    
    # add full-sibling
    for sib in sibs:
        for sib_ids in combinations(sib, 2):
            id1, id2 = sib_ids
            sex_1 = df_ped.loc[df_ped["offspring"] == id1, "sex"].values[0]
            sex_2 = df_ped.loc[df_ped["offspring"] == id2, "sex"].values[0]
            
            relation = "son_sister"
            if (sex_1 == "M") & (sex_2 == "M"):
                relation = "son_brother"
            elif (sex_1 == "F") & (sex_2 == "F"):
                relation = "daughter_sister"
            df_edges.loc[len(df_edges)] = [id1, id2, relation]
    
    return df_edges
    
def df2graph(df_edges):
    ids = list(set(df_edges["id1"].unique()) | set(df_edges["id2"].unique()))
    idx_id = pd.DataFrame({"idx": range(len(ids)), "id": ids})
    graph = [[0] * len(ids) for _ in range(len(ids))]
    for edge in df_edges[["id1", "id2"]].values:
        id1, id2 = edge
        idx1 = idx_id.loc[idx_id["id"] == id1, "idx"].values[0]
        idx2 = idx_id.loc[idx_id["id"] == id2, "idx"].values[0]
        
        graph[idx1][idx2] = 1
        graph[idx2][idx1] = 1
    
    return graph, idx_id
    
def get_shortest_path(graph, start, end):
    
    # paths = dfs.find_all_paths(copy.deepcopy(graph), start, end)
    # return shortest_list(paths)
    path = dfs.shortest_path(copy.deepcopy(graph), start, end)
    return path
    