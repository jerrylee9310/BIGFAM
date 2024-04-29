import numpy as np
import pandas as pd


def check_columns(
    df: pd.DataFrame, 
    column_types: dict, 
    column_components: dict = {}):
    """
    Check and reformat the columns of the DataFrame based on specified types and components.

    Parameters:
        df (pd.DataFrame): The DataFrame to be checked and reformatted.
        column_types (dict): A dictionary specifying the data type for each column.
        column_components (dict, optional): A dictionary specifying valid components for specific columns.

    Raises:
        KeyError: If any column specified in column_types is not found in the DataFrame.
        ValueError: If any column does not contain components specified in column_components.

    Returns:
        pd.DataFrame: The DataFrame after checking and reformatting columns.
    """
    # Check and reformat columns based on specified types
    for column, dtype in column_types.items():
        if column not in df.columns:
            raise KeyError(f"No '{column}' column found in the DataFrame.")
        
        df[column] = df[column].astype(dtype)
    
    # Check if columns contain valid components
    for column, components in column_components.items():
        if column not in df.columns:
            continue
        
        for unique_component in df[column].unique():
            if unique_component not in components:
                raise ValueError(f"Invalid component '{unique_component}' in column '{column}'. "
                                 f"Expected one of {components}.")
    
    return df

def get_unique_set(df, colns):
    
    return set(df[colns]
               .apply(lambda x: "_".join(map(str, sorted(x))),
                      axis=1))

def remove_duplicate_relpair(df: pd.DataFrame, coln_set: set):
    """
    Removes duplicate relative pairs from a DataFrame.

    This function identifies and removes rows representing duplicate 
    column name sets.
    
    Args:
        df (pd.DataFrame): The DataFrame containing relative pair information.
        coln_set (set): A set of column names representing the relative information used for identifying duplicates.

    Returns:
        pd.DataFrame: The DataFrame with duplicate relative pairs removed.
    """

    df = df.copy()
    df["set"] = (df[coln_set]
                 .apply(lambda x: "_".join(map(str, sorted(x))),
                        axis=1))
    df = df.drop_duplicates(subset="set", keep="first").drop(columns="set")
    
    return df

def flip_and_concat(df: pd.DataFrame, flip_colns: dict):
    flipping_flip_colns = dict((v,k) for k,v in flip_colns.items())
    flip_colns.update(flipping_flip_colns)
    
    
    df_original = df.copy()
    df_flipped = df.copy().rename(columns=flip_colns)
    
    return pd.concat([df_original, df_flipped], axis=0)