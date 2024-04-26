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