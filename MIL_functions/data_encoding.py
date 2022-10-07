import pickle
import blosc
import pandas as pd
import rdkit
import numpy as np
import pickle
import os
import sys
from rdkit import Chem
from rdkit.Chem import MACCSkeys,AllChem

def normalize_smiles(smi, InChi='NOT GIVEN'):      ## Converts each SMILES to an RDkit molecule then reconverts to SMILES. Ensures molecules with the same structure are the same SMILES
    try:
        smi_norm = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        return smi_norm
    except:
        try:
            smi_norm = Chem.MolToSmiles(Chem.MolFromInchi(InChi))
            return smi_norm
        except:
            return np.nan

def parent_finder(smi):         ## Searches the "smiles" column of the "data" dataframe of the original molecules and normalises both, returning a match if found
    for parent in data['smiles']:
        try:
            if Chem.MolToSmiles(Chem.MolFromSmiles(smi)) == Chem.MolToSmiles(Chem.MolFromSmiles(parent)):
                return parent
        except:
            continue
    return "No parent found"

def faster_parent_finder(smi):
    for parent in data['smiles']:
        try:
            if smi == parent:
                return parent
        except:
            continue
    return "No parent found" 

def drop_no_info_cols(df):
        res = df
        for col in df.columns:
                if len(df[col].unique()) == 1:
                        res = res.drop(col,axis=1)
                        print("dropped column:", col)
        return res

# def try_find_padel(x):
#     try:
#         padel = list(padel_data[padel_data['smiles']==x].drop(['Name','smiles'],axis=1).iloc[0])
#     except:
#         padel = np.nan
#     return padel

def get_ml_encoding(df, function=MACCSkeys.GenMACCSKeys):   ## returns a list, where each . Done to whole dataframe to allow for removing columns with NaN values
    def number_check(x):            ## Checks if a value can be converted to a float. Used to remove non-numeric encoding
        try:
            float(x)
            return x
        except:
            return "broken"

    ## Generate encoding list from smiles
    working_df = df.copy()      
    try:                     
        if function == 'PaDEL':
            # working_df['encoding_list'] = working_df['smiles'].apply(lambda x: try_find_padel(x)) 
            xxxx=1 ## useless line for if above is #'d out
        elif function == 'Morgan':
            working_df['encoding_list'] = working_df['smiles'].apply(lambda x: list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x),2)))   
        else:
            working_df['encoding_list'] = working_df['smiles'].apply(lambda x: list(function(Chem.MolFromSmiles(x))))   
    except:
        return np.nan
    ## Create dataframe from encoding list
    saved_df =working_df
    # working_df = working_df.dropna(subset='encoding_list')                                    
    ## Transform encoding dataframe into a list of lists. Can be used to generate a new column of the original dataframe                            
    X = saved_df['encoding_list'].to_list()                                                                                                                
    return X

def bag_parent(smiles,met_df,function):
    ## Create dataframe from molecules with parent == smiles as well as the smiles molecule. Removes duplicates
    mol_family          =   met_df[met_df["parent smiles"]==smiles].append({'smiles':smiles},ignore_index=True).drop_duplicates(subset=["smiles"])
    ##  Encodes this family using above
    mol_family_encoded  =   get_ml_encoding(df = mol_family, function = function)
    return mol_family_encoded

def create_compressed_pickle(data,path):
    pickled_data = pickle.dumps(data)  # returns data as a bytes object
    compressed_pickle = blosc.compress(pickled_data)

    with open(path, "wb") as f:
        f.write(compressed_pickle)

def load_compressed_pickle(path):
    with open(path, "rb") as f:
        compressed_pickle = f.read()

    depressed_pickle = blosc.decompress(compressed_pickle)
    data = pickle.loads(depressed_pickle)  # turn bytes object back into data
    return data