import pandas as pd
import numpy as np
import os
import pickle
from IPython.display import clear_output
from tpot import TPOTClassifier
from sklearn.feature_selection import VarianceThreshold
try:
    import misvm 
except:
    print("please use command to install MIL modelling package \n pip install -e git+https://github.com/garydoranjr/misvm.git#egg=misvm")
    quit()

def pos_or_neg(x):  ## Simple function used to translate predictions between MIL and ML into a single form
    if x>0:
        return 1
    else:
        return 0

def build_test_mil_model(training_data,testing_data,MIL,encoding,suffix,save_model,model_name = "",save_name="MIL_total_results.pk1"):    ## Build and test a MIL model
    already_complete = check_if_tested(suffix=suffix,encoding=encoding,model_name=model_name,path=save_name)
    if not already_complete:
        print("Building and testing:   fold:",suffix["fold"],"   Iteration:",suffix["iteration"],"   model:",model_name,"   encoding:",encoding)
        ##      Building model, note encoding already performed
        print('bagging');       bags = training_data[encoding+"_MIL"].to_list()
        print('labelling');     labels = training_data["Ames"].apply(lambda x: x if x==1 else -1).to_list()
        print('copying model'); model = MIL                                                              
        print('fitting');       model.fit(bags,labels)    
        print('BUILDING DONE, now testing');##      Testing model
        print('rebagging');     bags = testing_data[encoding+"_MIL"].to_list()
        print('relabelling');   labels = testing_data["Ames"].apply(lambda x: x if x==1 else -1).to_list()
        print('predicting');    predictions = model.predict(bags)                                      
        predicted_labels = list(map(pos_or_neg,predictions))                            
        df = pd.DataFrame({
            'predicted' : predictions,
            'predicted labal' : predicted_labels,
            'true label' : testing_data["Ames"].to_list()
        })  
        if type(save_name) == str:
            save_results(df = df, suffix = suffix, model = model_name, encoding = encoding, save_path = save_name)
        elif save_name:
            save_results(df = df, suffix = suffix, model = model_name, encoding = encoding)
        # if save_model:
        #     if suffix["fold"] == suffix["iteration"] == "":
        #         save_models(model = model, path = "OLD/saved_models/"+model_name+".sav")
        #     else:
        #         save_models(model = model, path = "/saved_models/"+model_name+"_"+str(suffix["fold"])+"_"+str(suffix["iteration"]+".sav"))
    else:
        print("Already tested   fold:",suffix["fold"],"   iteration:",suffix["iteration"],"   model:",model_name,"   encoding:",encoding)
        clear_output(wait=True)

def check_if_tested(suffix,model_name,encoding,path):    ## Checking if this build/test has already been done. Saves on time if a run crashes
    if not os.path.isfile(path): 
        already_complete = False
    else:
        results = pd.read_pickle(path)
        already_complete = ((results["fold"].isin([suffix["fold"]])) & (results["iteration"].isin([suffix["iteration"]])) & (results["model"].isin([model_name])) & (results["encoding"].isin([encoding]))).any()
    return already_complete

def format_results(df,suffix,model,encoding):  ## adds informative columns to the df used for saving results 
    df["fold"]  =   suffix["fold"]
    df["iteration"] =   suffix["iteration"]
    df['index'] = df.index
    df["model"] =   model
    df["encoding"] =   encoding
    return df

def save_results(df,suffix,model,encoding,save_path):     ## saves results to a single pickle, adding to it or generating it
    if not os.path.isfile(save_path):
        df_formatted = format_results(df=df,suffix=suffix,model=model,encoding=encoding)
        df_formatted.to_pickle(save_path)
    else:
        total_results = pd.read_pickle(save_path)
        df_formatted = format_results(df=df,suffix=suffix,model=model,encoding=encoding)
        total_results = pd.concat([total_results,df_formatted], ignore_index=True)
        total_results.to_pickle(save_path)
        
def save_models(model,path):                    ## Saves model to a path
    pickle.dump(model, open(path, 'wb'))

def develop_models(training_data,testing_data,suffix={"fold":"","iteration":""},encoding="MACCS",save_model=False,save_name="MIL_total_results.pk1",kernels=["linear", 'quadratic', 'polynomial']):     ## single function to complete whole pipeline for a set of data to all expected models
    ##      Step 0:     Checking that the encoding method described is expected
    if not encoding in ["MACCS","RDFP","Morgan"]:
        print('Please use expected encoding: ["MACCS", "RDFP"]')
        return
    print("encoding found")
    for kernel_type in kernels:
        ##      Step 1:     Build and test models
        print('developing model list')
        tested_mils =  [
            # ["MICA_"+kernel_type, misvm.MICA(kernel=kernel_type,verbose=False)],     
            ["MISVM_"+kernel_type, misvm.MISVM(kernel=kernel_type, C=1.0,verbose=False)],
            ['SIL_'+kernel_type, misvm.SIL(kernel=kernel_type,verbose=False)],
            # ["MissSVM_"+kernel_type,misvm.MissSVM(kernel=kernel_type,C=1.0,verbose=False)],
            ['NSK_'+kernel_type, misvm.NSK(kernel=kernel_type,verbose=False)],
            ["sbMIL-"+kernel_type, misvm.sbMIL(kernel=kernel_type,verbose=False)],
            ['sMIL_'+kernel_type, misvm.sMIL(kernel=kernel_type,verbose=False)]
            # ["STK_"+kernel_type, misvm.STK(kernel=kernel_type,verbose=False)]
            # ["stMIL_"+kernel_type, misvm.stMIL(kernel=kernel_type,verbose=False)]
                        ]
        ## note: Either as dask or non-dask TPOT can be used, defined in function variables
        for mil in tested_mils:
            print("built/test model:",mil[0])
            build_test_mil_model(training_data=training_data,testing_data=testing_data,suffix=suffix,MIL=mil[1],encoding=encoding,model_name=mil[0],save_model=save_model,save_name=save_name)

def build_test_ml_model(training_data,testing_data,encoding,suffix,save_name,save_model,tpot = False,splitting_name=''):                      ## Build and test a machine learning model
    already_complete = check_if_tested(suffix=suffix,encoding=encoding,model_name="TPOT",path=save_name)
    if not already_complete:
        print("Building and testing:   fold:",suffix["fold"],"   Iteration:",suffix["iteration"],"   model:","TPOT","   encoding:",encoding)
        ##      Building model, note encoding already performed
        instances = np.array(training_data[encoding].to_list())
        labels = training_data["Ames"].to_list() 
        if not tpot:   
            tpot_optimisation = TPOTClassifier(generations=10, population_size=500, cv=5, verbosity=0, n_jobs=8)
        else:
            tpot_optimisation = tpot
        tpot_optimisation.fit(instances,labels)    
        ##      Testing model
        model = tpot_optimisation.fitted_pipeline_  ## This takes the best fitted pipeline developed
        tpot_optimisation.export('tpot models/tpot_'+encoding+splitting_name+'_exported_pipeline.py')
        instances = np.array(testing_data[encoding].to_list())
        true_labels = testing_data["Ames"].to_list()       
        predictions = model.predict(instances) 
        predicted_probabilities = model.predict_proba(instances)                             
        predicted_labels = list(map(pos_or_neg,predictions))                            
        df = pd.DataFrame({
            'predicted' : [i[1] for i in predicted_probabilities],
            'predicted labal' : predicted_labels,
            'true label' : true_labels
        })   
        # save_results(df = df, suffix = suffix, model = "TPOT", encoding = encoding)  
        if type(save_name) == str:
            save_results(df = df, suffix = suffix, model = "TPOT", encoding = encoding, save_path = save_name)
        elif save_name:
            save_results(df = df, suffix = suffix, model = "TPOT", encoding = encoding)

        # if save_model:
        #     if suffix["fold"] == suffix["iteration"] == "":
        #         save_models(model = model, path = "OLD/saved_models/"+"TPOT"+".sav")
        #     else:
        #         save_models(model = model, path = "/saved_models/"+"TPOT"+"_"+str(suffix["fold"])+"_"+str(suffix["iteration"]+".sav"))
    else:
        print("Already tested   fold:",suffix["fold"],"   iteration:",suffix["iteration"],"   model:","TPOT","   encoding:",encoding)

def check_rank(df):
    count = 0
    for x in df['PaDEL_MIL']:
        matrix=np.array(x)
        if not np.linalg.matrix_rank(matrix) == len(x):
            count+=1
    if count != 0:
        print("This will fail...",count)

def normalise_PaDEL(df):
    def normalise_list_value(y,i,max_i,min_i):
        if max_i == min_i:
            y[i] = 0
        else:    
            y[i] = (y[i]-min_i)/(max_i-min_i)
        return y
    working = df.copy()
    for i in range(len(working['PaDEL_MIL'].iloc[0][0])):
        first = True
        for x in working['PaDEL_MIL']:
            for y in x:              
                if first:
                    first = False
                    max_i = y[i]; min_i = y[i]
                else:
                    max_i = max([y[i],max_i]); min_i = min([y[i],min_i])
        # print(max_i,min_i)
        working['PaDEL_MIL'] = working['PaDEL_MIL'].apply(lambda x: [normalise_list_value(y,i,max_i,min_i) for y in x])
    return working

def clean_data(df, remove_duplicates=False):
    def remove_duplicate_lists(x):
        duplicates = []; non_duplicates = []
        if type(x) == list:
            for i, test_list1 in enumerate(x):
                for j, test_list2 in enumerate(x):
                    if i < j:
                        if test_list1 == test_list2:
                            # print(i,'and',j,"are identical")
                            duplicates += [j]
            for i, list1 in enumerate(x):
                if i not in duplicates:
                    non_duplicates += [list1]
            return non_duplicates
        else:
            return np.nan
    working = df.copy()
    ## clean empty or NAN rows
    working = working.dropna(axis=0,how='any')
    ## Clean MIL duplicates
    if remove_duplicates:
        for MIL in ['MACCS_MIL','PaDEL_MIL']: 
            working[MIL] = working[MIL].apply(lambda x: remove_duplicate_lists(x))
    return working

def round_padel(df):
    def round_list(lst,dp):
        return [round(i,dp) for i in lst]

    # df['PaDEL'] = df['PaDEL'].apply(lambda x: round_list(x,5))
    df['PaDEL_MIL'] = df['PaDEL_MIL'].apply(lambda x: [round_list(lst,5) for lst in x])
    return df

def remove_zero_variance(inp,encoding='Morgan'):
    df = inp.copy()
    all_data = [lst for lists in df[encoding+'_MIL'].to_list() for lst in lists]
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(all_data)
    # df['Morgan'] = df['Morgan'].apply(lambda x: constant_filter.transform(np.array(x).reshape(1, -1)))
    df[encoding+'_MIL'] = df[encoding+'_MIL'].apply(lambda x: constant_filter.transform(x))
    return df
