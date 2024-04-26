# GOAL : 
# infer familialial relaitonship 
# with genetic correlation on X

import numpy as np
import pandas as pd
from BIGFAM import tools
from BIGFAM.pedigree import rel_rx
import importlib; importlib.reload(rel_rx)

def test(a):
    print(a)
    
relationships = ['SB', 'PC', 'AV', 'GP', '1C', 'HSB', 'HAV', 'GG']

def validate_input(
    df_ped: pd.DataFrame, 
    ids: list, 
    relationship: str
    ):
    
    # validate input data
    df_ped = (tools.check_columns(
        df=df_ped.copy(),
        column_types={
            "offspring": int, "sex": str, "father": int, "mother": int
        },
        column_components={"sex": ["M", "F"]})
        .reset_index(drop=True))
    assert len(ids) == 2
    assert relationship in relationships
    
    return df_ped, ids, relationship

def sort_id_by_age(df_pair: pd.DataFrame):
    df_sorted = pd.DataFrame()
    df_pair = tools.check_columns(
        df=df_pair.copy(),
        column_types={
            "eid": int,
            "eid_rel": int,
            "age": int,
            "age_rel": int
        }
    )
    
    for _, row in df_pair.iterrows():
        if row["age"] > row["age_rel"]:
            tmp = (pd.DataFrame(row).T
                   .rename(columns={
                       "eid": "eid_rel",
                       "age": "age_rel",
                       "eid_rel": "eid",
                       "age_rel": "age"
                   }))
        else:
            tmp = pd.DataFrame(row).T
        
        df_sorted = pd.concat([df_sorted, tmp])
    
    return df_sorted.reset_index(drop=True)
    
    
# DOR1 : ["PC", "SB"],
def get_rel_rx_dor1(
    df_ped: pd.DataFrame, 
    ids: list, 
    relationship: str
    ):
    
    # validate input data
    df_ped, ids, relationship = validate_input(
        df_ped=df_ped.copy(),
        ids=ids,
        relationship=relationship
    )
    
    # compute relative and rx
    rel_type, rx = None, np.nan
    
    if relationship == "PC":
        rel_type, rx = rel_rx.parent_offspring(df_ped, ids[0], ids[1])
        return rel_type, rx
    
    elif relationship == "SB":
        rel_type, rx = rel_rx.sibling(df_ped, ids[0], ids[1])
        return rel_type, rx
    
    else:
        raise Exception(
            f"'{relationship}' is not appropriate relationship name"
        )
        
# DOR2 : ["AV", "GP", "HSB"],
def get_rel_rx_dor2(
    df_ped: pd.DataFrame, 
    ids: list, 
    relationship: str
):
    # validate input data
    df_ped, ids, relationship = validate_input(
        df_ped=df_ped.copy(),
        ids=ids,
        relationship=relationship
    )
    
    # compute relative and rx
    rel_type, rx = None, np.nan
    
    if relationship == "AV":
        # rel_type, rx = rel_rx.avuncular(df_ped, ids[0], ids[1])
        return rel_type, rx
        
    
    elif relationship == "GP":
        return rel_type, rx
    
    elif relationship == "HSB":
        rel_type, rx = rel_rx.half_sibling(df_ped, ids[0], ids[1])
        return rel_type, rx
    
    else:
        raise Exception(
            f"'{relationship}' is not appropriate relationship name"
        )
        
# DOR3 : ["1C", "HAV", "GG"],