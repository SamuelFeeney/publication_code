import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from MIL_functions import data_splitting,model_building,data_encoding
from tpot import TPOTClassifier
from IPython.display import clear_output
import gc

try:
    import misvm 
except:
    print("please use command to install MIL modelling package \n pip install -e git+https://github.com/garydoranjr/misvm.git#egg=misvm")
    quit()

# Note: this  version of the model building code was implimented to due to the crashing of the older version when running TPOT. Crashes were due to RAM usage and this version limits the RAM available to the code.

############################
###### Memory limiter ######
############################

import sys
import warnings

import winerror
import win32api
import win32job




############################
######## MACCS KEYS ########
############################
def run_MACCS():
    input_data = [
        {'name':'random','function':data_splitting.random_split,'internal_save':'model_results\internal\MIL_aromatic_amine_cv_results.pk1','external_save':'model_results\external\ext_val_results.pk1'},
        {'name':'scaffold','function':data_splitting.scaffold_split,'internal_save':'model_results\internal\MIL_aromatic_amine_cv_Scaffold.pk1','external_save':'model_results\external\ext_val_results_scaffold_stratified.pk1'},
        {'name':'LSH','function':data_splitting.LSH,'internal_save':'model_results\internal\MIL_aromatic_amine_cv_LSH.pk1','external_save':'model_results\external\ext_val_results_LSH_stratified.pk1'},
        {'name':'SEC','function':data_splitting.SEC,'internal_save':'model_results\internal\MIL_aromatic_amine_cv_SEC.pk1','external_save':'model_results\external\ext_val_results_SEC_stratified.pk1'},
    ]


    data = data_encoding.load_compressed_pickle("data/encoded/encoded_data.dat")
    for splitting_method in input_data:
        ########## Internal Validation
        for encoding in ["MACCS"]:
            training_data,test_data = data_splitting.split_data(data,splitting_method['function'])
            rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=6234794)
            for fold,[train_index, validation_index] in enumerate(rskf.split(training_data, training_data["Ames"])):
                train   =   training_data.iloc[train_index];        validation    =   training_data.iloc[validation_index]
                model_building.develop_models(training_data=train,testing_data=validation,encoding = encoding,suffix={"fold":fold%10,"iteration":fold//10},save_model=False,save_name=splitting_method['internal_save'])
                del train; del validation; gc.collect()

        ########## External validation
        best_model = ["total_data_NSK_polynomial",misvm.NSK(kernel="polynomial",verbose=False)]; encoding = "MACCS"
        tpot_model = TPOTClassifier(generations=10, population_size=500, cv=5, verbosity=1, n_jobs=8)
        model_building.build_test_mil_model(training_data=training_data,testing_data=test_data,encoding=encoding,suffix={"fold":"","iteration":""},save_model=False,save_name=splitting_method['external_save'],model_name=best_model[0],MIL=best_model[1])
        model_building.build_test_ml_model( training_data=training_data,testing_data=test_data,encoding=encoding,suffix={"fold":"","iteration":""},save_model=False,save_name=splitting_method['external_save'],tpot=tpot_model,splitting_name=splitting_method['name'])

############################
########## Morgan ##########
############################
def run_Morgan():
    input_data = [
        {'name':'random','function':data_splitting.random_split,'internal_save':'model_results\internal\MIL_aromatic_amine_cv_results_MORGAN.pk1','external_save':'model_results\external\ext_val_results_MORGAN.pk1'},
        {'name':'scaffold','function':data_splitting.scaffold_split,'internal_save':'model_results\internal\MIL_aromatic_amine_cv_Scaffold_MORGAN.pk1','external_save':'model_results\external\ext_val_results_scaffold_stratified_MORGAN.pk1'},
        {'name':'LSH','function':data_splitting.LSH,'internal_save':'model_results\internal\MIL_aromatic_amine_cv_LSH_MORGAN.pk1','external_save':'model_results\external\ext_val_results_LSH_stratified_MORGAN.pk1'},
        {'name':'SEC','function':data_splitting.SEC,'internal_save':'model_results\internal\MIL_aromatic_amine_cv_SEC_MORGAN.pk1','external_save':'model_results\external\ext_val_results_SEC_stratified_MORGAN.pk1'},
    ]


    data = data_encoding.load_compressed_pickle("data/encoded/encoded_data.dat")
    data = model_building.remove_zero_variance(data,encoding='Morgan')
    for splitting_method in input_data:
        ########## Internal Validation
        for encoding in ["Morgan"]:
            training_data,test_data = data_splitting.split_data(data,splitting_method['function'])
            rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=6234794)
            for fold,[train_index, validation_index] in enumerate(rskf.split(training_data, training_data["Ames"])):
                train   =   training_data.iloc[train_index];        validation    =   training_data.iloc[validation_index]
                model_building.develop_models(training_data=train,testing_data=validation,encoding = encoding,suffix={"fold":fold%10,"iteration":fold//10},save_model=False,save_name=splitting_method['internal_save'])
                del train; del validation; gc.collect()

        ########## External validation
        best_model = ["total_data_NSK_polynomial",misvm.NSK(kernel="polynomial",verbose=False)]; encoding = "Morgan"
        tpot_model = TPOTClassifier(generations=10, population_size=500, cv=5, verbosity=3, n_jobs=-1)
        model_building.build_test_mil_model(training_data=training_data,testing_data=test_data,encoding=encoding,suffix={"fold":"","iteration":""},save_model=False,save_name=splitting_method['external_save'],model_name=best_model[0],MIL=best_model[1])
        model_building.build_test_ml_model( training_data=training_data,testing_data=test_data,encoding=encoding,suffix={"fold":"","iteration":""},save_model=False,save_name=splitting_method['external_save'],tpot=tpot_model,splitting_name=splitting_method['name'])

        ########################
        g_hjob = None

###
# memory limiter code


def create_job(job_name='', breakaway='silent'):
    hjob = win32job.CreateJobObject(None, job_name)
    if breakaway:
        info = win32job.QueryInformationJobObject(hjob,
                    win32job.JobObjectExtendedLimitInformation)
        if breakaway == 'silent':
            info['BasicLimitInformation']['LimitFlags'] |= (
                win32job.JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK)
        else:
            info['BasicLimitInformation']['LimitFlags'] |= (
                win32job.JOB_OBJECT_LIMIT_BREAKAWAY_OK)
        win32job.SetInformationJobObject(hjob,
            win32job.JobObjectExtendedLimitInformation, info)
    return hjob

def assign_job(hjob):
    global g_hjob
    hprocess = win32api.GetCurrentProcess()
    try:
        win32job.AssignProcessToJobObject(hjob, hprocess)
        g_hjob = hjob
    except win32job.error as e:
        if (e.winerror != winerror.ERROR_ACCESS_DENIED or
            sys.getwindowsversion() >= (6, 2) or
            not win32job.IsProcessInJob(hprocess, None)):
            raise
        warnings.warn('The process is already in a job. Nested jobs are not '
            'supported prior to Windows 8.')

def limit_memory(memory_limit):
    if g_hjob is None:
        return
    info = win32job.QueryInformationJobObject(g_hjob,
                win32job.JobObjectExtendedLimitInformation)
    info['ProcessMemoryLimit'] = memory_limit
    info['BasicLimitInformation']['LimitFlags'] |= (
        win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY)
    win32job.SetInformationJobObject(g_hjob,
        win32job.JobObjectExtendedLimitInformation, info)

def main():
    assign_job(create_job())
    memory_limit = 12 * 1024 * 1024 *1024 # 10 GB
    limit_memory(memory_limit)
    #run_MACCS()
    run_Morgan()
    try:
        bytearray(memory_limit)
    except MemoryError:
        print('Success: available memory is limited.')

    else:
        print('Failure: available memory is not limited.')
    return 0

if __name__ == '__main__':
    sys.exit(main())