import pandas as pd
from rdkit import Chem
from sklearn.model_selection import RepeatedStratifiedKFold
from MIL_functions import data_splitting
from rdkit.Chem import PandasTools
from sklearn.feature_selection import VarianceThreshold
from MIL_functions import model_building,data_encoding
## General functions
from MIL_functions import model_building
try:
    import misvm 
except:
    print("please use command to install MIL modelling package \n pip install -e git+https://github.com/garydoranjr/misvm.git#egg=misvm")
    quit()

def label_data(input_data,encodings,models,encoded_data_path="data/encoded/encoded_data_hansen.dat"):
    def remove_zero_variance(inp):
        df = inp.copy()
        all_data = [lst for lists in df['Morgan_MIL'].to_list() for lst in lists]
        constant_filter = VarianceThreshold(threshold=0)
        constant_filter.fit(all_data)
        # df['Morgan'] = df['Morgan'].apply(lambda x: constant_filter.transform(np.array(x).reshape(1, -1)))
        df['Morgan_MIL'] = df['Morgan_MIL'].apply(lambda x: constant_filter.transform(x))
        return df

    def get_labels(path):
        data = {} ## since padel data needs a little cleaning i generate two datasets held in a dictonary labled by the descriptor to be used upon
        data['MACCS'] = data_encoding.load_compressed_pickle(encoded_data_path)
        data['Morgan'] = data_encoding.load_compressed_pickle(encoded_data_path); data['Morgan'] = model_building.clean_data(data['Morgan']); data['Morgan'] = remove_zero_variance(data['Morgan'])
        rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=6234794)
        tested = {}
        for kernel in ['linear', 'polynomial']:
            mil = misvm.NSK(kernel=kernel,verbose=False)
            name = "NSK "+str(kernel)
            tested[name] = {}
            # for encoding in ["MACCS",'PaDEL','Morgan']:
            for encoding in ["MACCS",'Morgan']:
                dataset = data[encoding]
                tested[name][encoding] = []
                for fold,[train_index, validation_index] in enumerate(rskf.split(dataset, dataset["Ames"])):
                    validation    =   dataset.iloc[validation_index]
                    tested[name][encoding] += validation['smiles'].to_list()
        return tested
        
    outputs = {}
    labels = get_labels(encoded_data_path)
    for encoding in encodings:
        for model in models:
            label = labels[model][encoding]
            output_data = input_data.copy(); output_data=output_data[(output_data['encoding'] == encoding) & (output_data['model'] == model)].reset_index(drop=True)
            output_data['smiles'] = pd.Series(label)
            PandasTools.AddMoleculeColumnToFrame(output_data,smilesCol='smiles',molCol='Molecule')
            outputs[encoding+' '+model] = output_data
    return outputs
        
    outputs = {}
    labels = get_labels(encoded_data_path)
    for encoding in encodings:
        for model in models:
            label = labels[model][encoding]
            output_data = input_data.copy(); output_data=output_data[(output_data['encoding'] == encoding) & (output_data['model'] == model)].reset_index(drop=True)
            output_data['smiles'] = pd.Series(label)
            PandasTools.AddMoleculeColumnToFrame(output_data,smilesCol='smiles',molCol='Molecule')
            outputs[encoding+' '+model] = output_data
    return outputs

def molecule_group_analysis(data,function):
    molecule_group = data[data['Molecule'].apply(lambda x: function(x))]
    if molecule_group.empty:
        print('Molecule group not found')
        return
    else:
        print('Number of group:', len(molecule_group)/sum([len(molecule_group[molecule_group['encoding']==encoding]['model'].unique()) for encoding in molecule_group['encoding'].unique()])/10)
        amesP = sum(molecule_group['true label']==1)
        amesN = sum(molecule_group['true label']==0)
        print('Ames +:',amesP," "*(10-len(str(amesP))),'Ames -:',amesN," "*(10-len(str(amesN))),'Total:',amesN+amesP," "*(10-len(str(amesN))),'Ames + (%):',round(amesP*100/(amesN+amesP),3),)
        return molecule_group


## function checks
def polyaromatic_check(given_mol):
    mol = given_mol
    ri = mol.GetRingInfo()
    all_ring_bonds = ri.BondRings(); aromatic_ring_bonds=[]
    for ring_bonds in all_ring_bonds:
        if all([mol.GetBondWithIdx(idx).GetIsAromatic() for idx in ring_bonds]): ## check all bonds in ring are aromatic i.e. aromatic ring
            aromatic_ring_bonds += [ring_bonds]
    for ring_bonds1 in aromatic_ring_bonds:
        shared_aromatic_bonds = 0
        for bond in ring_bonds1:
            for ring_bonds2 in aromatic_ring_bonds:    ##compare bond to bonds in other rings
                if ring_bonds1 != ring_bonds2:
                    if bond in ring_bonds2:
                        shared_aromatic_bonds +=1
        if shared_aromatic_bonds >=2:       #as for any system with 3 fused rings there must be a central ring with two adjacent rings sharing a bond
            return True
    return False

def phenol_aniline_benzamide_check(given_mol):
    def one_benzene(molecule):
        benzenes = molecule.GetSubstructMatches(Chem.MolFromSmiles('c1ccccc1'))
        if len(benzenes) == 1:
            return True
        else:
            return False

    def benzamide_check(mole):
        benzamide = Chem.MolFromSmiles('NC(=O)c1ccccc1')
        if mol.HasSubstructMatch(benzamide):
            base_structures = mole.GetSubstructMatches(benzamide)
            for base_structure in base_structures:
                base_N = [base_atom for base_atom in [mole.GetAtomWithIdx(x)for x in base_structure] if base_atom.GetAtomicNum()==7]
                if len(base_N)==1:
                    base_N = base_N[0]
                    N_neighbors = [N_neighbor for N_neighbor in base_N.GetNeighbors() if N_neighbor.GetIdx() not in base_structure]
                    if N_neighbors:
                        if all(N_neighbor.GetAtomicNum() in [1,6] for N_neighbor in N_neighbors):
                            # neighbor_neighbors = [neighbor_neighbor for neighbor_neighbor in N_neighbors if neighbor_neighbor.GetIdx() not in base_structure]
                                # if all(neighbor_neighbor.GetAtomicNum() in [1,6] for neighbor_neighbor in neighbor_neighbors):
                            # if  all(all(neighbor_neighbor.GetAtomicNum() in [1,6] for neighbor_neighbor in N_neighbor.GetNeighbors() if neighbor_neighbor else True) for N_neighbor in N_neighbors if N_neighbor else True):
                                if  all(all(neighbor_neighbor.GetAtomicNum() in [1,6] if (neighbor_neighbor and neighbor_neighbor.GetIdx() not in base_structure) else True for neighbor_neighbor in N_neighbor.GetNeighbors()) for N_neighbor in N_neighbors):
                                    return True
                    else:
                        return True
        
    mol = given_mol
    mols_to_check = {
        'phenol'    :   Chem.AddHs(Chem.MolFromSmiles('Oc1ccccc1'), onlyOnAtoms=[atom.GetIdx() for atom in Chem.MolFromSmiles('Oc1ccccc1').GetAtoms() if atom.GetAtomicNum() == 8]),
        'analine'   :   Chem.AddHs(Chem.MolFromSmiles('Nc1ccccc1'), onlyOnAtoms=[atom.GetIdx() for atom in Chem.MolFromSmiles('Nc1ccccc1').GetAtoms() if atom.GetAtomicNum() == 7]),
        'benzamide' :   Chem.MolFromSmiles('NC(=O)c1ccccc1')
    }
    mol = Chem.AddHs(mol)
    if one_benzene(mol):
        for molecule in mols_to_check.values(): 
            if molecule == mols_to_check['benzamide']:
                if benzamide_check(mol):
                    return True  
            else: 
                if mol.HasSubstructMatch(molecule):
                    return True
    return False

def alkyl_alkenyl_halide_check(given_mol):  
    if given_mol:
        mol = given_mol
        mol = Chem.AddHs(mol)

        halides = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9,17,35,53,85,117]]
        if any(halides):
            ring_info = mol.GetRingInfo()
            if not ring_info.NumRings() >0:
                if all(True if atom.GetAtomicNum() in [1,6,9,17,35,53,85,117] else False for atom in mol.GetAtoms()):
                    return True


        halides = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9,17,35,53,85,117]]
        # if any(halides):
        #     halides = [halide for halide in halides if halide.GetDegree()==1]
        #     if any(halides):
        #         for halide in halides:
        #             halide_C_neighbors = [neighbor for neighbor in halide.GetNeighbors() if (neighbor.GetAtomicNum()==6 and neighbor.GetIsAromatic()==False and not neighbor.IsInRing())]
        #             if any(halide_C_neighbors):
        #                 halide_C_C_bonds = [bond.GetBondType() for bond in halide_C_neighbors[0].GetBonds() if all([bond.GetEndAtom().GetAtomicNum()==6,bond.GetBeginAtom().GetAtomicNum()==6])]
        #                 if any(halide_C_C_bonds):
        #                     if any([halide_C_C_bond in [Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE] for halide_C_C_bond in halide_C_C_bonds]):
        #                         return True
    return False

def four_benzyl_piperidines_check(given_mol):
    mol = given_mol
    group = Chem.MolFromSmiles('*N2CCC(Cc1ccccc1)CC2')
    if mol.HasSubstructMatch(group):
        return True
    return False

def azide_check(given_mol):
    mol = given_mol
    group = Chem.MolFromSmiles('[N-]=[N+]=N')
    if mol.HasSubstructMatch(group):
        return True
    return False

def octahydrobenzo_f_benzoquinolines_check(given_mol):
    mol = given_mol
    group = Chem.MolFromSmiles('*c1cccc2c1CCC3C2CC(*)CN3*')
    if mol.HasSubstructMatch(group):
        return True
    return False

def aromatic_nitro_check(given_mol):
    mol = given_mol
    ###### Complex aromatic check
    # nitro = Chem.MolFromSmiles('O=NO')
    # if mol.HasSubstructMatch(nitro):
    #         base_structures = mol.GetSubstructMatches(nitro)
    #         for base_structure in base_structures:
    #             base_N = [base_atom for base_atom in [mol.GetAtomWithIdx(x)for x in base_structure] if base_atom.GetAtomicNum()==7]
    #             if len(base_N)==1:
    #                 base_N = base_N[0]
    #                 N_neighbors = [N_neighbor for N_neighbor in base_N.GetNeighbors() if N_neighbor.GetIdx() not in base_structure]
    #                 if len(N_neighbors)==1:
    #                     N_neighbor = N_neighbors[0]
    #                     if N_neighbor.GetIsAromatic() == True:
    #                         return True

    ###### Simple aromatic check (benzene)
    aromatic_nitro = Chem.MolFromSmiles('O=[N+](O)c1ccccc1')
    if mol.HasSubstructMatch(aromatic_nitro):
        return True
    return False

            # Note: simple aromatic check was used since the complex version found more molecules than that in the paper using a subset of its data

def quinoline_check(given_mol):
    failed = False
    mol = given_mol
    quinoline = Chem.MolFromSmiles('c2ccc1ncccc1c2')
    if mol.HasSubstructMatch(quinoline): 
        ###### This checks that no additional aromatic groups are attached to the quinoline (extra fused rings)
        base_structures = mol.GetSubstructMatches(quinoline)
        for base_structure in base_structures:
            for idx in base_structure:
                base = mol.GetAtomWithIdx(idx)
                neighbors = [neighbor for neighbor in base.GetNeighbors() if neighbor.GetIdx() not in base_structure]
                if any(neighbor.GetIsAromatic() for neighbor in neighbors):
                    failed = True
            if not failed:
                return True
    return False

            # Note: simple aromatic check was used since the complex version found more molecules than that in the paper using a subset of its data

def furan_wo_nitro_check(given_mol):
    def check_mol_nitrogen(mole,nitrogen_index,furan_atom_index):
        edited_mol = Chem.RWMol(mole)
        nitro = Chem.MolFromSmiles('O=NO')
        furan_N_bond_idx = edited_mol.GetBondBetweenAtoms(nitrogen_index,furan_atom_index).GetIdx()
        frags = Chem.FragmentOnBonds(edited_mol, [furan_N_bond_idx])
        # try:
        global f
        if len(Chem.GetMolFrags(frags, asMols=False)) >1:
            for frag in Chem.GetMolFrags(frags, asMols=True):
                if frag.HasSubstructMatch(nitro):
                    if any(all(True for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum()==7 and len(atom.GetNeighbors())==1) for atom in frag.GetAtoms() if atom.GetAtomicNum()==0):
                        if frag.GetNumAtoms()==4:
                            return False
            return True
    def exclude_furan_fused_rings(mole):
        furan = Chem.MolFromSmiles('c1ccoc1')
        if mole.HasSubstructMatch(furan):
            base_structures = mole.GetSubstructMatches(furan)
            for base_structure in base_structures:
                for idx in base_structure:
                    base = mole.GetAtomWithIdx(idx)
                    neighbors = [neighbor for neighbor in base.GetNeighbors() if neighbor.GetIdx() not in base_structure]
                    if neighbors:
                        if not any(mole.GetBondBetweenAtoms(idx,neighbor.GetIdx()).IsInRing() for neighbor in neighbors):
                            return True
        return False
    
    mol = given_mol
    furan = Chem.MolFromSmiles('c1ccoc1')
        
    if exclude_furan_fused_rings(mol):
        if mol.HasSubstructMatch(furan):
            base_structures = mol.GetSubstructMatches(furan)
            for base_structure in base_structures:
                failed = False
                for idx in base_structure:
                    base = mol.GetAtomWithIdx(idx)
                    neighbors = [neighbor for neighbor in base.GetNeighbors() if neighbor.GetIdx() not in base_structure]
                    if neighbors:
                        N_neighbor = [neighbor for neighbor in neighbors if neighbor.GetAtomicNum()==7]
                        if len(N_neighbor):
                            N_neighbor = N_neighbor[0]
                            if not check_mol_nitrogen(mole=mol,nitrogen_index = N_neighbor.GetIdx(),furan_atom_index=idx):
                                    failed = True

                if not failed:
                    return True
                    
    return False