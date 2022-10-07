from cProfile import label
from scipy.spatial import ConvexHull
from scipy import interpolate
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from scipy.integrate import simps
from numpy import trapz
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.decomposition import PCA

def pos_or_neg(x):  ## Simple function used to translate predictions between MIL and ML into a single form
    if x>0:
        return 1
    else:
        return 0

def pickle_transform(pickle):
    rslt_list = []
    results = pickle
    results["true label"] = results["true label"].apply(lambda x: pos_or_neg(x))
    for iteration in results["iteration"].unique():
        for fold in results["fold"].unique():
            for model in results["model"].unique():
                for encoding in results["encoding"].unique():
                    working_data = results[(results["fold"]==fold)&(results["iteration"]==iteration)&(results["model"]==model)&(results["encoding"]==encoding)]
                    [TP,TN,FP,FN] = confusion_matrix(working_data)
                    rslt_list += [{"encoding":encoding, "model":model, "fold":fold, "iteration":iteration, "TP":TP, "TN":TN, "FP":FP, "FN":FN}]
    return pd.DataFrame(rslt_list)

def confusion_matrix(df):
    TP = len(df[(df["predicted labal"] == 1) & (df["true label"] == 1)])
    TN = len(df[(df["predicted labal"] == 0) & (df["true label"] == 0)])
    FP = len(df[(df["predicted labal"] == 1) & (df["true label"] == 0)])
    FN = len(df[(df["predicted labal"] == 0) & (df["true label"] == 1)])
    return [TP,TN,FP,FN]

def PCA_plot_mean_line_hull(input,total_data,names=False,colours = False):
    if len(colours) != len(input):
        print("colour list isn't the same length as the inputs")
        colours = False
    pca = PCA(n_components=2)
    pca.fit([x for data in input for x in data['MACCS'].to_list()])
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)

    total_mean  = [sum(x)/len(x) for x in list(map(list, zip(*pca.transform(total_data['MACCS'].to_list()))))]
    for i, data in enumerate(input):
        mapped_data = pca.transform(data['MACCS'].to_list())
        mean = [sum(x)/len(x) for x in list(map(list, zip(*mapped_data)))]


        hull = ConvexHull(mapped_data)
        x_hull = np.append(mapped_data[hull.vertices,0],
                        mapped_data[hull.vertices,0][0])
        y_hull = np.append(mapped_data[hull.vertices,1],
                        mapped_data[hull.vertices,1][0])
        
        # interpolate
        dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
        dist_along = np.concatenate(([0], dist.cumsum()))
        spline, u = interpolate.splprep([x_hull, y_hull], 
                                        u=dist_along, s=0)
        interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
        interp_x, interp_y = interpolate.splev(interp_d, spline)
        # plot shape
        if colours:
            plt.fill(interp_x, interp_y, '--', alpha=0.1,c= colours[i])
            ax.scatter(mapped_data[:,0], mapped_data[:,1],c= colours[i],s=10)
            ax.plot([total_mean[0],mean[0]], [total_mean[1],mean[1]],c= colours[i])
        else:
            plt.fill(interp_x, interp_y, '--', alpha=0.1)
            ax.scatter(mapped_data[:,0], mapped_data[:,1],s=10)
            ax.plot([total_mean[0],mean[0]], [total_mean[1],mean[1]])
    
    if names:
        ax.legend(names)
    else:
        ax.legend(['data '+str(i//3) for i in range(len(input)*3)])

def macro_mirco_mean_stdv(data,ext_val=False):
    def calc_metrics(data):
        working = data.copy()
        def row_metrics(row):
            def my_accuracy(row):
                if not (row["TP"]+row["TN"]+row["FP"]+row["FN"]) == 0:
                    acc = (row["TP"]+row["TN"])/(row["TP"]+row["TN"]+row["FP"]+row["FN"])
                else:
                    acc = "No results"
                return acc
            def my_cohen_kappa(row): 
                if not ((row['TP']+row['FP'])==0 or (row['FP']+row['TN'])==0) or not ((row['TP']+row['FN'])==0 or (row['FN']+row['TN'])==0):
                    cohen_kappa = ((row['TN']*row['TP'])-(row['FN']*row['FP']))/(((row['TP']+row['FP'])*(row['FP']+row['TN']))+((row['TP']+row['FN'])*(row['FN']+row['TN'])))
                else:
                    cohen_kappa = "No results"
                return cohen_kappa
            def my_sensitivity(row):
                if not (row["TP"]+row["FP"]) == 0:
                    sens = row["TP"]/(row["TP"]+row["FP"])
                else:
                    sens = "No positive predictions"
                return sens
            def my_specificity(row):
                if not (row["TN"]+row["FN"]) == 0:
                    spec = row["TN"]/(row["TN"]+row["FN"])
                else:
                    spec = "No negative predictions"
                return spec
            def my_F1(row):
                if not (2*row["TP"]+row["FP"]+row["FN"]) == 0:
                    f1 = (2*row["TP"])/(2*row["TP"]+row["FP"]+row["FN"])
                else:
                    f1 = "Only True Negatives"
                return f1
            return [my_accuracy(row),my_sensitivity(row),my_specificity(row),my_F1(row),my_cohen_kappa(row)]
        working[['accuracy','sensitivity','specificity','F1','Cohen Kappa']]= working.apply(lambda x: row_metrics(x),axis=1, result_type='expand')
        return working

    working = data.copy(); macro_results = {}; micro_results = {}
    working = calc_metrics(working)

    def fail_safe_devide(numerator,denominator):
        if denominator == 0:
            return 0
        else:
            return numerator/denominator

    def macro(df):
        ######## NOTE: removed models if TP+FP=0 or TN+FN=0 ########
        df = df[(df['TP']!=0) | (df['FP']!=0)]
        df = df[(df['TN']!=0) | (df['FN']!=0)]
        ############################################################
        results = {
            'F1':               fail_safe_devide(sum(df['F1']),len(df)),
            'F1_stdv':          np.std(df['F1']),
            'Cohen Kappa':      fail_safe_devide(sum(df['Cohen Kappa']),len(df)),
            'Cohen Kappa_stdv': np.std(df['Cohen Kappa']),
            'accuracy':         fail_safe_devide(sum(df['accuracy']),len(df)),
            'accuracy_stdv':    np.std(df['accuracy']),
            'sensitivity':      fail_safe_devide(sum(df['sensitivity']),len(df)),
            'sensitivity_stdv': np.std(df['sensitivity']),
            'specificity':      fail_safe_devide(sum(df['specificity']),len(df)),
            'specificity_stdv': np.std(df['specificity'])
        }
        return results


    def micro(df):
        TP = sum(df['TP']); TN = sum(df['TN']); FP = sum(df['FP']); FN = sum(df['FN'])
        results = {
            'F1':               2*TP/(2*TP+FN+FP),
            'Cohen Kappa':      ((TN*TP)-(FN*FP))/(((TP+FP)*(FP+TN))+((TP+FN)*(FN+TN))),
            'CK2':              ((TP+TN)-(((FN+TN)*(TP+FP)+(FN+TN)*(FP+TN))/(TP+TN+FP+FN)))/(((TP+TN+FN+FP)-(((FN+TN)*(TP+FP)+(FN+TN)*(FP+TN))/(TP+TN+FP+FN)))),
            'accuracy':         (TP+TN)/(TP+TN+FP+FN),
            'sensitivity':      (TP/(TP+FP) if not TP+FP ==0 else 0),
            'specificity':      (TN/(TN+FN) if not TN+FN ==0 else 0),
        }
        return results   
    

    for model in working['model'].unique():
        for encoding in working['encoding'].unique():
            global working_model
            working_model = working[(working['model']==model)&(working['encoding']==encoding)]
            if working_model.empty or len(working_model)!=100:
                if not ext_val:
                    print(encoding,model,'is missing data')

            micro_results[encoding+" "+model] = micro(working_model)
            if not ext_val:
                macro_results[encoding+" "+model] = macro(working_model)
            
    micro_df = pd.DataFrame.from_dict(micro_results,orient = 'index'); 
    micro_df[['encoding','model']] = [i.split(' ',1) for i in micro_df.index]; 
    micro_df = micro_df.reset_index(drop=True); micro_df = micro_df[micro_df.columns.tolist()[-2:]+micro_df.columns.tolist()[:-2]]
    if ext_val:
        return micro_df
    macro_df = pd.DataFrame.from_dict(macro_results,orient = 'index'); 
    macro_df[['encoding','model']] = [i.split(' ',1) for i in macro_df.index]; 
    macro_df = macro_df.reset_index(drop=True); macro_df = macro_df[macro_df.columns.tolist()[-2:]+macro_df.columns.tolist()[:-2]]
    return [macro_df,micro_df]

def ext_val_metrics(data):
    return macro_mirco_mean_stdv(data,ext_val=True)

def my_grouped_bar(result_dict,metric='accuracy',cmaps=['Oranges','Greens','Purples','Reds','Yellows'],ext=False,grouping_size=5):
    labels = list(result_dict.keys()); values = {}; encodings = 0
    fig, ax = plt.subplots(figsize=(15, 8))
    for encoding in list(result_dict.values())[0]['encoding'].unique():
        values[encoding] = {model: [results[(results['model']==model)&(results['encoding']==encoding)][metric].values[0] for results in result_dict.values()] for model in result_dict[labels[0]]['model'].unique()}
    x = np.arange(len(labels))  # the label locations
    # width = 0.1  # the width of the bars
    if ext:
        width = 1/((len(list(values.values())[0])*len(values))+.5)
    else:
        width = 1/((len(list(values.values())[0])*len(values))+1.5)

    rects = []
    # for j,encoding in enumerate(values):
    for j,encoding in enumerate(values):
        for i,model in enumerate(values[encoding]):
            position = width*(i+j*2) -width*((len(list(values.values())[0])*len(values))-1)/2
            if len(values) >1:
                name = model.replace('_',' ').replace('-',' ')+' '+str(encoding)
            else:
                name = model.replace('_',' ').replace('-',' ')
            if ext:
                cmap = plt.cm.get_cmap('Set2')
                rect = ax.bar(x + position, values[encoding][model], width, label=name,color=cmap(i))
            else:
                cmap = plt.cm.get_cmap(cmaps[(i+j*2)//(grouping_size)])
                rect = ax.bar(x + position, values[encoding][model], width, label=name,color=cmap(((i+j*2)%(grouping_size)+1)*(1/6)))
            rects += [rect]

    ax.set_ylabel(metric)
    ax.set_xticks(x, labels)
    ax.set_xlim([-width*((len(list(values.values())[0])*len(values))+1)/2, len(labels)+width*((len(list(values.values())[0])*len(values))+1)/2-1])
    ax.set_ylim([0, 1])
    if ext:
        ax.legend(fancybox=True)
    else:
        ax.legend(loc='upper right', bbox_to_anchor=(1.17, 1), ncol=1, fancybox=True)
    plt.show()
    return 

def AUROC_analysis(data_dict,num_step=25,cmaps=['Oranges','Greens','Purples','Blues'],specific_models=False,line_thickness=1):
    plt.figure(figsize=(10, 10))
    plt.plot([0,1],[0,1], 'r--', label='random')
    colour_index=0
    for key in data_dict:
        data = data_dict[key]
        for model_name in specific_models if specific_models else data['model'].unique():
            for encoding in data['encoding'].unique():
                sens_points[encoding] = pd.DataFrame();spec_points[encoding] = pd.DataFrame()
                b = data[(data["model"] == model_name) & (data["encoding"] == encoding)]
                if not b.empty:
                    mini  = min(b["predicted"])
                    maxi  = max(b["predicted"])
                    diff  = maxi-mini
                    steps = [mini + i*diff/(num_step-1) for i in range(num_step)]

                    sensitivity = {}; specificity = {}; inv_spec = {}
                    for i,step in enumerate(steps):
                        name = str(i+1)+" step"
                        b[name] = 0
                        b.loc[b['predicted'] > step, name] = 1
                        
                        TP = len(b[(b[name] == 1) & (b["true label"] == 1)])
                        TN = len(b[(b[name] == 0) & (b["true label"] == 0)])
                        FP = len(b[(b[name] == 1) & (b["true label"] == 0)])
                        FN = len(b[(b[name] == 0) & (b["true label"] == 1)])

                        if not TP+FN == 0:
                            sensitivity[name] = TP/(TP+FN)
                        else:
                            sensitivity[name] = 0

                        if not TN+FP == 0:
                            specificity[name] = TN/(TN+FP)
                            inv_spec[name] = 1-(TN/(TN+FP))
                        else:
                            specificity[name] = 0
                            inv_spec[name] = 1

                    # AUROC_trapizoidal   = np.trapz(x=list(inv_spec.values()), y= list(sensitivity.values()))
                    # print(key,encoding,model_name.replace('_',' ').replace('-',' '),"Trapizoidal rule:",' '*(40-len(key)-len(model_name)-len(encoding)),round(-AUROC_trapizoidal,3))
                    if cmaps:
                        cmap = plt.cm.get_cmap(cmaps[colour_index//2])
                        plt.plot(inv_spec.values(), sensitivity.values(),c=cmap((colour_index%2+1)/3),label=str(key)+" "+str(encoding)+" "+str(model_name.replace('_',' ').replace('-',' ')),linewidth=line_thickness)
                        colour_index +=1 
                    else:
                        plt.plot(inv_spec.values(), sensitivity.values(),label=str(key)+" "+str(encoding)+" "+str(model_name.replace('_',' ').replace('-',' ')),linewidth=line_thickness)

    plt.legend(bbox_to_anchor=(1.17, 1), ncol=1, fancybox=True)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()
    # print(".: sklearn method seems to use the trapizoidal rule method")

def get_AUROC(data_dict, paste_dict,num_step=25):
    output_dict = {}
    for key in data_dict:
        data = data_dict[key]
        paste = paste_dict[key]
        AUROCs = []
        for model_name in data['model'].unique():
            for encoding in data['encoding'].unique():
                b = data[(data["model"] == model_name) & (data["encoding"] == encoding)]
                if not b.empty:
                    mini  = min(b["predicted"])
                    maxi  = max(b["predicted"])
                    diff  = maxi-mini
                    steps = [mini + i*diff/(num_step-1) for i in range(num_step)]

                    sensitivity = {}; specificity = {}; inv_spec = {}
                    for i,step in enumerate(steps):
                        name = str(i+1)+" step"
                        b[name] = 0
                        b.loc[b['predicted'] > step, name] = 1
                        
                        TP = len(b[(b[name] == 1) & (b["true label"] == 1)])
                        TN = len(b[(b[name] == 0) & (b["true label"] == 0)])
                        FP = len(b[(b[name] == 1) & (b["true label"] == 0)])
                        FN = len(b[(b[name] == 0) & (b["true label"] == 1)])

                        if not TP+FN == 0:
                            sensitivity[name] = TP/(TP+FN)
                        else:
                            sensitivity[name] = 0

                        if not TN+FP == 0:
                            specificity[name] = TN/(TN+FP)
                            inv_spec[name] = 1-(TN/(TN+FP))
                        else:
                            specificity[name] = 0
                            inv_spec[name] = 1

                    AUROCs += [{'encoding':encoding, 'model':model_name, 'AUROC': round(-np.trapz(x=list(inv_spec.values()), y= list(sensitivity.values())),5)}]
        output_dict[key] = paste.merge(right=pd.DataFrame(AUROCs),on=['model','encoding'])
    return output_dict

def range_finder(x1,y1,x2,y2):
    deltaX = (max(x1) - min(x1))/5
    deltaY = (max(y1) - min(y1))/5
    xmin1 = min(x1) - deltaX
    xmax1 = max(x1) + deltaX
    ymin1 = min(y1) - deltaY
    ymax1 = max(y1) + deltaY

    deltaX = (max(x2) - min(x2))/5
    deltaY = (max(y2) - min(y2))/5
    xmin2 = min(x2) - deltaX
    xmax2 = max(x2) + deltaX
    ymin2 = min(y2) - deltaY
    ymax2 = max(y2) + deltaY
    return [min([xmin2,xmin1]),max([xmax1,xmax2])],[min([ymin2,ymin1]),max([ymax1,ymax2])]

def density_map_data(x,y,x_range,y_range):
    xmin,xmax = x_range
    ymin,ymax = y_range
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    return xx,yy,f

## note, sizes given in cm, will be converted to inches for usage in matplotlib
def density_PCA(Xt,name,size,font,x_range,y_range):
    def cm_in(num):
        return num/2.54
    figure_size = tuple(cm_in(i) for i in size)
    print(figure_size)
    fig, ax = plt.subplots(figsize=figure_size)
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(1, 1, 1, figsize = figure_size)
    x,y,z = density_map_data(Xt[:,0],Xt[:,1],x_range,y_range)
    ax.contour(x,y,z,cmap='plasma')
    if name:
        ax.set_title(name+' (n='+str(len(Xt[:,0]))+')')
    ax.set_xlabel('PC1',fontsize=10)
    ax.set_ylabel('PC2',fontsize=10)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    return fig

def get_ROC_points(data_dict,num_step=25,specific_models=False):
    sens_points = {};spec_points = {}
    for encoding in ["MACCS",'Morgan']:
        sens_points[encoding] = pd.DataFrame();spec_points[encoding] = pd.DataFrame()
        for key in data_dict:
            data = data_dict[key]
            for model_name in specific_models if specific_models else data['model'].unique():
                b = data[(data["model"] == model_name) & (data["encoding"] == encoding)]
                if not b.empty:
                    mini  = min(b["predicted"])
                    maxi  = max(b["predicted"])
                    diff  = maxi-mini
                    steps = [mini + i*diff/(num_step-1) for i in range(num_step)]

                    sensitivity = {}; specificity = {}; inv_spec = {}
                    for i,step in enumerate(steps):
                        name = str(i+1)+" step"
                        b[name] = 0
                        b.loc[b['predicted'] > step, name] = 1
                        
                        TP = len(b[(b[name] == 1) & (b["true label"] == 1)])
                        TN = len(b[(b[name] == 0) & (b["true label"] == 0)])
                        FP = len(b[(b[name] == 1) & (b["true label"] == 0)])
                        FN = len(b[(b[name] == 0) & (b["true label"] == 1)])

                        if not TP+FN == 0:
                            sensitivity[name] = TP/(TP+FN)
                        else:
                            sensitivity[name] = 0

                        if not TN+FP == 0:
                            specificity[name] = TN/(TN+FP)
                            inv_spec[name] = 1-(TN/(TN+FP))
                        else:
                            specificity[name] = 0
                            inv_spec[name] = 1
                    sens_points[encoding][model_name+"_"+key+"_sensitivity"]= sensitivity.values()
                    spec_points[encoding][model_name+"_"+key+"_specificity"]= inv_spec.values()
    return list(sens_points.values())+list(spec_points.values())
    # print(".: sklearn method seems to use the trapizoidal rule method")