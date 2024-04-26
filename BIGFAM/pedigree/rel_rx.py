import pandas as pd
import numpy as np

def test(ss):
    print(ss)
    
dict_rel_rx = {
    "son-father": ["SF", 0],
    "son-mother": ["SM", 1/np.sqrt(2)],
    "daughter-father": ["DF", 1/np.sqrt(2)],
    "daughter-mother": ["DM", 1/2],
    "son-brother": ["SB", 1/2],
    "son-sister": ["SS_DB", 1/np.sqrt(8)],
    "daughter-sister": ["DS", 3/4],
}

def find_parent(
    df_ped: pd.DataFrame, 
    off_id: int
    ):
    df_po = df_ped[df_ped["offspring"] == off_id]
    father_id, mother_id = df_po[["father", "mother"]].values[0]
    return (father_id, mother_id)

def find_sibling(df_parent: pd.DataFrame, fid: int, mid: int):
    df_off = df_parent[(df_parent["father"] == fid) 
                       & (df_parent["mother"] == mid)]
    sib_ids = df_off["volid"].values
    return sib_ids
    
def get_sex(df_ped, eid):
    if eid in list(df_ped["offspring"]):
        return df_ped.loc[df_ped["offspring"] == eid, "sex"].values[0]
    else:
        if eid in list(df_ped["father"]):
            return "M"
        elif eid in list(df_ped["mother"]):
            return "F"
    
def parent_offspring(df_ped, id1, id2):
    
    for _, row in df_ped.iterrows():
        id1 == row["offspring"]
        off_sex = row["sex"]
        
        if id2 == row["father"]:
            if off_sex == "M":
                return dict_rel_rx["son-father"]
            elif off_sex == "F":
                return dict_rel_rx["daughter-father"]
        elif id2 == row["mother"]:
            if off_sex == "M":
                return dict_rel_rx["son-mother"]
            elif off_sex == "F":
                return dict_rel_rx["daughter-mother"]
        else:
            pass
            
            
def sibling(df_ped, id1, id2):
    ids = [id1, id2]
    tmp = df_ped[df_ped["offspring"].isin(ids)]
    
    # validate
    assert len(tmp) == 2, f"full sibling {ids} has more than 2 columns."
    
    sex_1 = tmp.loc[tmp["offspring"] == id1, "sex"].values[0]
    sex_2 = tmp.loc[tmp["offspring"] == id2, "sex"].values[0]
    
    if (sex_1 == "M") & (sex_2 == "M"):
        return dict_rel_rx["son-brother"]
    
    if (sex_1 == "F") & (sex_2 == "F"):
        return dict_rel_rx["daughter-sister"]

    return dict_rel_rx["son-sister"]

def half_sibling(df_ped, id1, id2):
    sex_1 = get_sex(df_ped, id1)
    sex_2 = get_sex(df_ped, id2)
    father_id1, mother_id1 = find_parent(df_ped, id1)
    father_id2, mother_id2 = find_parent(df_ped, id2)
    
    # paternal
    if father_id1 == father_id2:
        if (sex_1 == "M") & (sex_2 == "M"):
            rel_type, rx = "SB_fatherSame", 0
        elif (sex_1 == "F") & (sex_2 == "F"):
            rel_type, rx = "DS_fatherSame", 1/2
        else:
            rel_type, rx = "SS_DB_fatherSame", 0
        
        return rel_type, rx
    
    # maternal
    if mother_id1 == mother_id2:
        if (sex_1 == "M") & (sex_2 == "M"):
            rel_type, rx = "SB_motherSame", 1/2
        elif (sex_1 == "F") & (sex_2 == "F"):
            rel_type, rx = "DS_motherSame", 1/4
        else:
            rel_type, rx = "SS_DB_motherSame", 1/np.sqrt(8)
        
        return rel_type, rx
    
    

def avuncular(df_ped, id1, id2):
    
    # maternal
    
    # paternal
    pass