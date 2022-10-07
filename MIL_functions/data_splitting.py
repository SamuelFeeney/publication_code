# https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00576-2
from rdkit.Chem.Scaffolds import MurckoScaffold, rdScaffoldNetwork
from sklearn.model_selection import train_test_split
from rdkit import Chem,DataStructs
from rdkit.Chem import MACCSkeys
import random
import pandas as pd

def split_data(data,splitting_function):
    train_1,test_1 = splitting_function(data[data['Ames']==1])
    train_0,test_0 = splitting_function(data[data['Ames']==0])
    training_data = pd.concat([train_0,train_1]); test_data = pd.concat([test_0,test_1])
    training_data = training_data.reset_index(drop=True);   test_data = test_data.reset_index(drop=True)
    return training_data,test_data

def random_split(data):
    train, test = train_test_split(data, test_size=0.2, random_state=34783)
    return train, test

def scaffold_split_old(data):
    working = data.copy()
    def get_scaffold(mol):
        params = rdScaffoldNetwork.ScaffoldNetworkParams()
        net = rdScaffoldNetwork.CreateScaffoldNetwork([mol],params)
        # nodemols = [Chem.MolFromSmiles(x) for x in net.nodes]
        return [x for x in net.nodes]
    def get_num_rings(mols):
        num_rings = {}
        for mol in mols:
            num = Chem.rdMolDescriptors.CalcNumRings(mol)
            if num in num_rings:
                num_rings[num] +=1
            else:
                num_rings[num] =1
        return num_rings

    working['scaffold'] = working['smiles'].apply(lambda x: get_scaffold(Chem.MolFromSmiles(x)))
    scaffolds = set([x for scafs in working['scaffold'] for x in scafs])
    ring_dict = get_num_rings([Chem.MolFromSmiles(x) for x in scaffolds])

    def select_on_scaffold(scaffold,selection,ring_num,test_size):
        global test_count
        if selection == 1:
            return 1
        if test_count < test_size:
                if any([1 for smi in scaffold if Chem.rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(smi))==ring_num]):
                    test_count +=1
                    return 1
        return 0

    global test_count
    working['test'] = 0; test_count = 0; test_size=len(working)//5
    for ring_num in sorted([i for i in ring_dict],reverse=True):
        working['test'] = working.apply(lambda row: select_on_scaffold(row['scaffold'],row['test'],ring_num,test_size), axis=1)
    test = working[working['test']==1]
    train = working[working['test']==0]
    return train,test
    
def scaffold_split(data):
    working = data.copy()
    def get_scaffold(mol):
        params = rdScaffoldNetwork.ScaffoldNetworkParams()
        net = rdScaffoldNetwork.CreateScaffoldNetwork([mol],params)
        # nodemols = [Chem.MolFromSmiles(x) for x in net.nodes]
        return [x for x in net.nodes]
    def closest_to_n_rings(smis,N=3):
        num_rings = {}
        for smi in smis:
            num = Chem.rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(smi))
            if num in num_rings:
                num_rings[num] += [smi]
            else:
                num_rings[num] = [smi]
        min_dist = min([abs(n-N) for n in num_rings])
        ring_sizes = [n for n in num_rings if abs(n-N)==min_dist]
        values = []
        for ring_size in ring_sizes:
            values += num_rings[ring_size]
        if len(values) ==1:
            return values[0]
        else:
            return sorted(values)[0]
    def test_select(scaffold,selection,selected_scaffold):
        global test_count
        if selection == 1:
            return 1
        if test_count < test_size:
                if scaffold == selected_scaffold:
                    test_count +=1
                    return 1
        return 0

    working['scaffold network'] = working['smiles'].apply(lambda x: get_scaffold(Chem.MolFromSmiles(x)))
    working['scaffold'] = working['scaffold network'].apply(lambda x: closest_to_n_rings(x))

    test_size = len(working)//5
    working['test'] = 0
    global test_count; test_count=0; random.seed(34783)
    while test_count < test_size:
        scaffold_order = random.sample(list(working['scaffold'].unique()),len(working['scaffold'].unique()))
        # print(scaffold_order)
        for selected_scaffold in scaffold_order:
            working['test'] = working.apply(lambda x:test_select(x['scaffold'],x['test'],selected_scaffold),axis=1)
    test = working[working['test']==1]
    train = working[working['test']==0]
    return train, test

def LSH(data,num_desc_freq=16):
    working = data.copy()
    def get_desc_freqs(descs):
        desc_len = len(descs.iloc[0])
        freqs = {}
        for i in range(desc_len):
            desc_i = descs.apply(lambda x: x[i])
            freqs[i] = sum(desc_i)/len(desc_i)
        return freqs
    desc_freqs = get_desc_freqs(working['MACCS'])
    desc_freqs = {i: abs(val-0.5) for i,val in desc_freqs.items()}
    desc_freqs = dict(sorted(desc_freqs.items(), key=lambda item: item[1]))
    desc_freqs = {i: desc_freqs[i] for i in list(desc_freqs)[:num_desc_freq]}

    def get_bin_desc(desc,desc_freqs):
        bin_desc = {}
        for i in desc_freqs:
            bin_desc[i] = desc[i]
        return tuple(bin_desc.values())
    working['bin_desc'] = working['MACCS'].apply(lambda x: get_bin_desc(x,desc_freqs))

    working['bin'] = "not assigned"
    for i,bin in enumerate(working['bin_desc'].unique()):
        working.loc[working['bin_desc'] == bin, 'bin'] = i 


    def test_select(bin,selection,selected_bin):
        global test_count
        if selection == 1:
            return 1
        if test_count < test_size:
                if bin == selected_bin:
                    test_count +=1
                    return 1
        return 0  
    test_size = len(working)//5
    working['test'] = 0
    global test_count; test_count=0; past_bins = []; random.seed(34783)
    while test_count < test_size:
        selected_bin = random.randrange(0,len(working['bin'].unique())-1)
        if selected_bin not in past_bins:
            past_bins += [selected_bin]
            working['test'] = working.apply(lambda x:test_select(x['bin'],x['test'],selected_bin),axis=1)
    test = working[working['test']==1]
    train = working[working['test']==0]
    return train, test

def SEC(data,tc=0.6):
    working = data.copy()
    global clusters; clusters = {}
    mol_order = list(working.index)
    random.seed(34783)
    random.shuffle(mol_order)
    working = working.loc[mol_order]

    def get_cluster(smi):
        desc = Chem.MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smi))
        distance_between = {}
        global clusters
        if clusters:
            for cluster in clusters:
                distance_between[cluster] = DataStructs.FingerprintSimilarity(desc,clusters[cluster], metric=DataStructs.TanimotoSimilarity)

            closest_cluster = min(distance_between, key=distance_between.get)
            if distance_between[closest_cluster] <= tc:
                return closest_cluster
        
        clusters[len(clusters)+1] = desc
        return len(clusters)+1
    working['cluster'] = ""
    working['cluster'] = working['smiles'].apply(lambda x: get_cluster(x))

    def test_select(cluster,selection,selected_cluster):
        global test_count
        if selection == 1:
            return 1
        if test_count < test_size:
                if cluster == selected_cluster:
                    test_count +=1
                    return 1
        return 0  
    test_size = len(working)//5
    working['test'] = 0
    global test_count; test_count=0; past_cluster = []; random.seed(34783)
    while test_count < test_size:
        selected_cluster = random.randrange(0,len(working['cluster'].unique())-1)
        if selected_cluster not in past_cluster:
            past_cluster += [selected_cluster]
            working['test'] = working.apply(lambda x:test_select(x['cluster'],x['test'],selected_cluster),axis=1)
    test = working[working['test']==1]
    train = working[working['test']==0]
    return train, test