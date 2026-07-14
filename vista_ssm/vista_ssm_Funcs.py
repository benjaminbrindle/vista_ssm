import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn
import random
from sklearn.metrics import confusion_matrix
from time import process_time
import pickle
from scipy.optimize import linprog
import sklearn.datasets
import networkx as nx

from vista_ssm import EMmlgssm, InitEMmlgssm, EMlgssm

def savedic(dic,loc):
    """saves a dictionary to a pickle file at a given location

    Parameters
    ----------
    dic: dictionary
        Dictionary to save
    loc : str
        The file location

    """
    with open(loc, 'wb') as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

def loaddic(loc):
    """loads a dictionary from a pickle file at a given location

    Parameters
    ----------
    loc : str
        The file location

    Returns
    -------
    dictionary
        saved dictionary
    """
    with open(loc, 'rb') as f:
        return pickle.load(f)
    
def lgssmSample(params,time_points):
    """generates data generated from a LGSSM with the given parameters

    Parameters
    ----------
    params : dic
        contains the following:
            A : np.ndarray(dim_x,dim_x)

            C: np.ndarray(dim_y,dim_x)

            mu: np.ndarray(dim_x,1)

            Gamma: np.ndarray(dim_x,dim_x)

            Sigma: np.ndarray(dim_y,dim_y)

            P: np.ndarray(dim_x,dim_x)
    
    time_points: ndarray(n_time)
        times corresponding to each observation 
        
    Returns
    -------
    np.ndarray(n_time,dim_y,1)
        data sampled from lgssm
    """
    diffs = time_points[1:]-time_points[:-1]
    T = np.insert(diffs,0,diffs[0])
    dx=len(params['A'])
    dy=len(params['Sigma'])
    x=params['mu']+np.random.multivariate_normal(np.zeros(dx),params['P']).reshape((dx,1))
    Y=(params['C']@x+np.random.multivariate_normal(np.zeros(dy),params['Sigma']/T[0]).reshape((dy,1))).T
    for t in range(1,len(T)):
        x=(params['A']*T[t]+np.eye(dx))@x+np.random.multivariate_normal(np.zeros(dx),params['Gamma']*T[t]).reshape((dx,1))
        Y=np.vstack([Y,(params['C']@x+np.random.multivariate_normal(np.zeros(dy),params['Sigma']/T[t]).reshape((dy,1))).T])    
    return Y.reshape((len(T),dy,1))

def mlgssmSample(N_list,params,T,varT,T_diff=2,**kwargs):
    """generates data generated from a MLGSSM with the given parameters

    Parameters
    ----------
    N_list : list of ints of len n_cluster
        number of samples to draw from each lgssm
    params : dic
        contains the following:
            A : list of n np.ndarray(dim_x,dim_x)

            C : list of n np.ndarray(dim_y,dim_x)

            mu : list of n np.ndarray(dim_x,1)

            Gamma : list of n np.ndarray(dim_x,dim_x)

            Sigma : list of n np.ndarray(dim_y,dim_y)

            P : list of n np.ndarray(dim_x,dim_x)
    
    T: int
        Number of desired time steps in returned data  
    varT: bool
        Determines if the number of time steps should be chosen randomly   
    T_diff: int, optional
        Determines distribution of time step chosen in variable step case

    time_points: ndarray(n_samples,n_time), optional
        times corresponding to each observation in dataset, if not specified assumes uniform sampling period

    Returns
    -------
    np.ndarray(n_samples,n_time,dim_y,1)
        data sampled from lgssm
    np.ndarray(n_samples,n_time)
        times for each observation of sampled data
    np.ndarray(n_samples,1)
        labels of data
    """

    N=sum(N_list)
    if kwargs:
        if 'time_points' in kwargs:
            time_points = kwargs['time_points']
        else:
            time_points=[]
            for n in range(N):
                if varT:
                    individual_T=random.randint(T-T_diff,T+T_diff)
                else:
                    individual_T=T
                time_points.append(np.linspace(0,1,individual_T))
    else:
        time_points=[]
        for n in range(N):
            if varT:
                individual_T=random.randint(T-T_diff,T+T_diff)
            else:
                individual_T=T
            time_points.append(np.linspace(0,1,individual_T))
    
    data=[]
    count=0
    for k in range(len(N_list)):
        param_cluster = dict([(key,params[key][k]) for key in params])
        for n in range(N_list[k]):
            data.append(lgssmSample(param_cluster,time_points[count]))
            count+=1
    if varT:
        data=np.array(data, dtype='object')
    else:
        dy=len(params['Sigma'][0])
        data=np.array(data).reshape(N, T, dy, 1)
    label=np.array([i for i in range(len(N_list)) for k in range(N_list[i])])
    indices = np.arange(N)
    np.random.shuffle(indices)
    label=label[indices]
    data=data[indices]
    time_points=time_points[indices]
    return (data,time_points,label)


#MLGSSM initialization
def initializationmethod(how,param_dic,dataset,time_points):
    """generates initial parameters for EMLGSSM algorithm

    Parameters
    ----------
    how: bool
        which method to use for initializing the parameters:
            random: chooses random parameters within a specific range
            ident: chooses parameters as identity matrices or close to them
            kmeans: uses kmeans on the data to choose parameters
    param_dic: dic
        parameters used in initialization:
            DIM_X
            DIM_Y
            N_CLUSTER
            NUM_CPU
            FIX
            NUM_LGSSM
    dataset: ndarray(n_samples,n_time,dim_y,1), optional for kmeans initialization

    time_points: ndarray(n_samples,n_time), optional for kmeans initialization
        times corresponding to each observation in dataset

    Returns
    -------
    dic
        dictionary of lists of parameters. size of list depends on 'N_CLUSTER' in param_dic
    """
    dx=param_dic['DIM_X']
    dy=param_dic['DIM_Y']
    n_cluster=param_dic['N_CLUSTER']
    if how=='random':
        A_list=[np.diag(np.random.uniform(-1.9,-0.1,dx)) for n in range(n_cluster)]
#         C_list=[np.vstack([np.ones((1,dx)),np.random.rand(dy-1,dx)]) for n in range(n_cluster)]
        C_list=[np.vstack([np.ones((1,dx)),np.random.randint(2, size=(dy-1, dx))]) for n in range(n_cluster)]
        mu_list=[np.random.rand(dx,1) for n in range(n_cluster)]
        gamma_list=[sklearn.datasets.make_spd_matrix(dx) for n in range(n_cluster)]
        sigma_list=[sklearn.datasets.make_spd_matrix(dy) for n in range(n_cluster)]
        P_list=[sklearn.datasets.make_spd_matrix(dx) for n in range(n_cluster)]
        return {'mu':np.array(mu_list),'P':np.array(P_list), 'A':np.array(A_list),
                'Gamma':np.array(gamma_list),'C':np.array(C_list),'Sigma':np.array(sigma_list), 
                'weight': np.ones(n_cluster)/n_cluster}
    elif how=='ident':
        A_list=[-1.5*np.eye(dx) for n in range(n_cluster)]
        if dx>=dy:
            C_list=[np.hstack([np.hstack([np.eye(dy) for i in range(dx//dy)]),np.eye(dy)[:,:dx%dy-dy]]) for n in range(n_cluster)]
        else:
            C_list=[np.vstack([np.vstack([np.eye(dx) for i in range(dy//dx)]),np.eye(dx)[:dy%dx-dx,:]]) for n in range(n_cluster)]
        mu_list=[x*np.ones((dx,1)) for x in np.linspace(-1,1,n_cluster)]
        gamma_list=[np.eye(dx)*0.1 for n in range(n_cluster)]
        sigma_list=[np.eye(dy)*0.1 for n in range(n_cluster)]
        P_list=[np.eye(dx)*0.1 for n in range(n_cluster)]
        return {'mu':np.array(mu_list),'P':np.array(P_list), 'A':np.array(A_list),
                'Gamma':np.array(gamma_list),'C':np.array(C_list),'Sigma':np.array(sigma_list), 
                'weight': np.ones(n_cluster)/n_cluster}
    elif how=='kmeans':
        init_em = InitEMmlgssm(
            n_clusters=n_cluster, 
            dim_x=dx, 
            dim_y=dy, 
            n_cpu=param_dic['NUM_CPU']
        )

        return init_em.fit_tuning( 
            Y=dataset,
            time_points=time_points,
            fix_param=param_dic['FIX'], 
            n_lgssm=param_dic['NUM_LGSSM']
        )
    

def runVISTA(how,param_dic,dataset,time_points,**kwargs):
    """helper function to run EMLGSSM algorithm

    Parameters
    ----------
    how: string
        which method to use for initializing the parameters:
            random: chooses random parameters within a specific range
            ident: chooses parameters as identity matrices or close to them
            kmeans: uses kmeans on the data to choose parameters
    param_dic: dic
        parameters used:
            DIM_X
            DIM_Y
            N_CLUSTER
            NUM_CPU
            FIX - list of parameters to be fixed throughout algorithm
            NUM_LGSSM - number of lgssms to use in kmeans initialization for each time series
            MAX_ITER
            EPSILON
            BIC - bool of whether or not to return bayesian information criterion
            
    dataset: ndarray(n_samples,n_time,dim_y,1)

    time_points: ndarray(n_samples,n_time)
        times corresponding to each observation in dataset
    
    inits: dic, optional
        dictionary of lists of parameters to initialize the algorithm directly
        
    labels: ndarray(n_samples), optional
        ground truth labels to compare classification performance to; if provided, prints information

    save : bool, optional
        If True, save parameters after each iteration to the file location given by loc.
        
    loc : string, optional
        File location to save parameters.
           
    Returns
    -------
    dic
        dictionary of lists of parameters outputted by algorithm, and other metrics
    """
    start=process_time()
    if kwargs:
        if 'inits' in kwargs:
#             expects input of structure {'mu':np.array(mu_list),'P':np.array(P_list), 'A':np.array(A_list),
#                 'Gamma':np.array(gamma_list),'C':np.array(C_list),'Sigma':np.array(sigma_list), 
#                 'weight': np.ones(n_cluster)/n_cluster}
            init_params = kwargs['inits']
        else:
            init_params = initializationmethod(how,param_dic,dataset,time_points)

        if 'save' in kwargs and 'loc' in kwargs:
            save=kwargs['save']
            loc=kwargs['loc']
        else:
            save=False
            loc=''
    else:
        init_params = initializationmethod(how,param_dic,dataset,time_points)
        
    model = EMmlgssm(
        state_mats=init_params["A"], 
        state_covs=init_params["Gamma"], 
        obs_mats=init_params["C"], 
        obs_covs=init_params["Sigma"], 
        init_state_means=init_params["mu"], 
        init_state_covs=init_params["P"], 
        weights=init_params["weight"]
    )

    result = model.fit(
        Y=dataset, 
        time_points=time_points,
        max_iter=param_dic['MAX_ITER'], 
        epsilon=param_dic['EPSILON'], 
        n_cpu=param_dic['NUM_CPU'], 
        fix_param=param_dic['FIX'],
        bic=param_dic['BIC'],
        save=save,
        loc=loc
    )
    result['runtime']=process_time()-start
    if 'labels' in kwargs:
        labels = kwargs['labels']
        cm = confusion_matrix(labels, result['label'].reshape(param_dic['NUM_DATA']))
        print('\nConfusion matrix')
        print(cm)
        print('\n')
        print('Runtime (cpu)')
        print(result['runtime'])
        print('\n')
        print('Information Criterions (BIC,ABIC,AIC,AICc)')
        print(result['bic'])
    return result
        
def mlgssmPlots(result,ll=True,tol=True,toltail=True,params=True):
    """plots outputs from running the mlgssm algorithm for visualization

    Parameters
    ----------
    result: dictionary
        output from mlgssm algorithm
    ll : bool, optional
        whether or not to plot the log-likelihood
    tol : bool, optional
        whether or not to plot the stopping tolerance 
    toltail : bool, optional
        whether or not to plot the tail of the stopping tolerance separately (after the tenth iteration)
    params : bool, optional
        whether or not to plot the model parameters using plt.show
        
    """
    
    if ll:
        for i in range(len(result['parameter']['A'])):
            plt.plot(np.array(result['likelihood'])[:,i])
            plt.title('Log-likelihood, cluster '+str(i))
            plt.show()    
    if tol:
        plt.plot(np.array(result['tolerance']))
        plt.title('Stopping tolerance')
        plt.show()    
    if toltail:
        plt.plot(np.arange(10,len(result['tolerance'])),np.array(result['tolerance'])[10:])
        plt.title('Stopping tolerance')
        plt.show()
    if params:
        fig, ax = plt.subplots(6, len(result['parameter']['A']),figsize=(15,25))
        for l, param in enumerate([x for x in result['parameter']][1:]):
            for k in range(len(result['parameter']['A'])):
                im = ax[l, k].matshow(result['parameter'][param][k])
                plt.colorbar(im, ax=ax[l, k])
                ax[l,k].set_title(param+'_'+str(k))
        plt.show()
        
        
#https://stackoverflow.com/questions/48511584/how-to-efficiently-make-class-to-cluster-matching-in-order-to-calculate-resultin
def best_perm(label_true,label_predicted,m,par=None):
    """computes the permutation matrix which will permute the columns of a computed confusion matrix in such a way as to maximize the trace

    Parameters
    ----------
    label_true: np.ndarray(n,)
        The ground truth labels

    label_predicted: np.ndarray(n,)
        The predicted labels to be mapped to the original labels by the permutation matrix

    m: int
        Number of clusters for the problem at hand

    par: dictionary, optional
        A dictionary of parameters returned from the MLGSSM algorithm to be mapped from the predicted labels to the original labels

    Returns
    -------
    Dictionary:
        confusion_martix : confusion matrix - np.ndarray(n,n)
        permuation_matrix : optimal permutation matrix - np.ndarray(n,n)
        label : optimal labels - np.ndarray(n,)
        map : dictionary mapping true and predicted labels - dictionary
        parameter : dictionary of remapped parameters - dictionary
    """

    cm=confusion_matrix(label_true,label_predicted)
    n, n = cm.shape
    #for loop addresses the case when you may be expecting m clusters but an algorithm returned less than m clusters
    for x in [x for x in range(m) if x not in label_true and x not in label_predicted]:
        cm = np.hstack([np.hstack([cm[:,0:x],np.zeros((n,1)),cm[:,x:]])])
        cm = np.vstack([np.vstack([cm[0:x,:],np.zeros((1,n+1)),cm[x:,:]])])
        n, n = cm.shape
    res = linprog(-cm.ravel(),
                  A_eq=np.r_[np.kron(np.identity(n), np.ones((1, n))),
                             np.kron(np.ones((1, n)), np.identity(n))],
                  b_eq=np.ones((2*n,)), bounds=n*n*[(0, None)])
    assert res.success
    pm=res.x.reshape(n, n).T
    map_labels=np.argmax(pm,axis=1)
    labels_dic=dict([(i,map_labels[i]) for i in range(len(map_labels))])
    results={'confusion_matrix':cm,'permutation_matrix':pm,'label':np.vectorize(labels_dic.get)(label_predicted),'map':labels_dic}
    if par:
        results['parameter']=dict([(key,np.array([par[key][labels_dic[x]] for x in range(len(par[key]))])) for key in par])
    return results

def summarystats(label_true,label_predicted,m):
    """prints some summary statistics when given true and predicted labels

    Parameters
    ----------
    label_true: np.ndarray(n,)
        ground truth labels
    label_predicted: np.ndarray(n,)
        predicted labels
    m: int
        expected number of classes
    """
    results=best_perm(label_true,label_predicted,m)
    cm=results['confusion_matrix']
    pm=results['permutation_matrix']
    labels_pred=results['label']
    
    print('confusion matrix')
    print(cm@pm.astype('int'))
    cm1=cm@pm
    total1=sum(sum(cm1))
    accuracy1=(sum(np.diag(cm1)))/total1
    print ('Accuracy : ', accuracy1)
    print('Adjusted Rand Index : ', sklearn.metrics.adjusted_rand_score(label_true,labels_pred))
    if len(cm)==2:
        print('ROC AUC: ', sklearn.metrics.roc_auc_score(label_true,labels_pred))
    # else:
    #     print('ROC AUC: ', sklearn.metrics.roc_auc_score(labels,labels_pred,multi_class='ovr'))
    print('\n')
    print(sklearn.metrics.classification_report(label_true,labels_pred))

def noiseless_trajectory(params,T,T_final):
    """computes and returns an array of time series determined from given parameters, setting noise equal to zero

    Parameters
    ----------
    params: dictionary
        A dictionary of parameters for a MLGSSM
    T: int
        Number of observations in the returned time series
    T_final: float
        Sets end time of time series, assuming time series begins at t=0

    Returns
    -------
    np.ndarray(n_clusters,T,n_features,1)
        Noiseless time series for each of the LGSSMs (n_clusters in all) in the MLGSSM
    """
    nc=len(params['weight'])
    sim_dics=[dict([(key,params[key][k]) for key in params]) for k in range(nc)]
    for k in range(nc):
        for covmat in ['P','Gamma','Sigma']:
            sim_dics[k][covmat]=np.zeros(sim_dics[k][covmat].shape)
    return np.array([lgssmSample(sim_dics[k],np.linspace(0,T_final,T)) for k in range(nc)])
    
def predicted_trajectories(params,data,label,T,T_final=1,num_sam=3,legend=False,xplot=[0,1],plotcolor=(plt.cm.rainbow,0,1,1),**kwargs):
    """Plots noiseless trajectories from a dictionary of MLGSSM parameters overlaid with observed data and (optionally) an additional noiseless trajectory corresponding to the MLGSSM that generated the data

    Parameters
    ----------
    params: dictionary
        A dictionary of parameters for a MLGSSM
    data: np.ndarray(n_samples,n_observations,n_features,1)
        Observed data corresponding to the given MlGSSM - n_observations can vary among samples
    label: np.ndarray(n_samples,)
        Labels mapping each observed time series to a given cluster
    T: int
        Number of samples in the returned time series
    T_final: float (optional)
        Sets end time of time series, assuming time series begins at t=0. Default assumes time series is sampled from [0,1]
    num_sam: int (optional)
        How many randomly selected time series corresponding to each cluster (via the label parameter) to plot. If num_sam > # time series, all the time series will be plotted
    legend: boolean (optional)
        If True shows names for the time series plotted in the legend. Default False
    xplot: list of length 2
        Start and end points to form a linspace for the x-axis of the plots if 0-1 range is not desired for a particular visualization 
    plotcolor: tuple (colormap,min_color,max_color,alpha)
        Tuple for manually setting properties of the plot. Default (plt.cm.rainbow,0,1,1)
    features: list (optional)
        List of length n_features corresponding to the feature names to be plotted
    true: dictionary (optional)
        A dictionary of parameters for a MLGSSM, assumed to be the true parameters that generated the given data
    row_labels: list (optional)
        List of length n_samples corresponding to names for the given time series
    ylim: list (optional)
        List of form [ymin, ymax] setting the bounds of the plots
    timepoints: np.ndarray(n_samples,nobservations) (optional)
        Observed data observation timestamps  - n_observations can vary among samples. If not provided, assumes uniform sampling on xplot.
    """

    dy=len(params['Sigma'][0])
    nc=len(params['weight'])
    n=len(data)
    if kwargs:
        if 'features' in kwargs:
            features=kwargs['features']
        else:
            features=['Feature '+str(i) for i in range(dy)]
        if 'row_labels' in kwargs:
            row_labels=kwargs['row_labels']
        else:
            row_labels=['Series '+str(i) for i in range(n)]
        if 'true' in kwargs:
            true_params=kwargs['true']
            true_X=noiseless_trajectory(true_params,T,T_final)
        if 'timepoints' in kwargs:
            tp=kwargs['timepoints']
        else:
            tp=np.array([np.linspace(xplot[0],xplot[1],len(d)) for d in data],dtype=object)
    else:
        features=['Feature '+str(i) for i in range(dy)]
        tp=np.array([np.linspace(xplot[0],xplot[1],len(d)) for d in data],dtype=object)
     
    sim_X=noiseless_trajectory(params,T,T_final)

    num_sam_list=[min(num_sam,list(label).count(k)) for k in range(nc)]

    real_ind=[random.sample(list(np.arange(0,len(data),1)[np.where(label==k)]),num_sam_list[k]) for k in range(nc)]
    
    fig = plt.figure(figsize=(15,3*dy))
    for i, feat in enumerate(features):
        for k in range(nc):
            ax=fig.add_subplot(dy,nc,nc*i+k+1)
            for l in range(num_sam_list[k]):
                X=data[real_ind[k][l]]
                if legend:
                    ax.plot(tp[real_ind[k][l]],X[:,i],label=row_labels[real_ind[k][l]],alpha=plotcolor[3])
                else:
                    ax.plot(tp[real_ind[k][l]],X[:,i],label='_nolegend_',alpha=plotcolor[3])
            for l,j in enumerate(ax.lines):
                colormap = plotcolor[0]
                colors = [colormap(i) for i in np.linspace(plotcolor[1], plotcolor[2], len(ax.lines))]
                j.set_color(colors[l])
            ax.plot(np.linspace(xplot[0],xplot[1],len(sim_X[k][:,i])),sim_X[k][:,i],'r--',label='Predicted Trajectory',linewidth=2)
            if 'true' in kwargs:
                ax.plot(np.linspace(xplot[0],xplot[1],len(true_X[k][:,i])),true_X[k][:,i],'k:',label='True Trajectory',linewidth=2)
            ax.legend()
            if 'ylim' in kwargs:
                ax.set_ylim(kwargs['ylim'])
            ax.set_title(f'{feat}, Cluster {k}')
    plt.show()

def agg_perf(loc,clus,dims,post=1,std=[True,True],criteria=list(range(4)),label=np.empty(0)):
    """Prints tables of the information criteria, cluster similarity, and ARI for given runs of mlgssm algorithm

    Parameters
    ----------
    loc: String
        Where to find saved pickle dictionaries from mlgssm algorithm
    clus: List
        List of ints of the number of clusters to check
    dims: List
        List of ints of the latent dimensions to check
    post: int 
        How  many runs of the algorithm for each (n_cluster,dim_x) pair to compute
    std: List
        First position = True prints standard deviation of information criterion, second position = True prints standard deviation of accuracy and ARI
    criteria: List 
        Default [0,1,2,3] corresponding to printing all of BIC, ABIC, AIC, AICc, subsets of this list will print only the corresponding criteria
    label: np.ndarray
        Ground truth labels for the problem in question
    """
    ic=[]
    ac=[]
    ari=[]
    crit=['BIC','ABIC','AIC','AICc']
    for dx in dims:
        for nc in clus:
            for i in range(post):
                r=loaddic(f'{loc}{nc}_cluster_{dx}_latent_{i}.pickle')
                ic.append(r['bic'])
                if len(r['label'].shape)!=1:
                    r['label']=r['label'].reshape(-1)
                if label.size > 0:
                    result_permuted=best_perm(label,r['label'],nc)
                    cm=result_permuted['confusion_matrix']
                    pm=result_permuted['permutation_matrix']
                    cmp=cm@pm
                    total=sum(sum(cmp))
                    accuracy=(sum(np.diag(cmp)))/total
                    ac.append(accuracy)
                    ari.append(sklearn.metrics.adjusted_rand_score(label,r['label']))
    n=len(clus)
    d=len(dims)
    
    info_mean=np.array([np.mean(np.array(ic)[post*i:post*(i+1)],axis=0) for i in range(n*d)])
    info_std=np.array([np.std(np.array(ic)[post*i:post*(i+1)],axis=0) for i in range(n*d)])
        
    print('Information Criteria:')
    for c in criteria:
        print('\n'+crit[c]+':')
        if std[0]:
            info=np.array([f'{info_mean[i,c]:.2E}'+f' ({info_std[i,c]:.2f})' for i in range(n*d)]).reshape((d,n))
        else:
            info=np.array([f'{info_mean[i,c]:.2E}' for i in range(n*d)]).reshape((d,n))
        print(pd.DataFrame(data=info,columns=clus,index=dims))
    if label.size > 0:
        ac_mean=[np.mean(np.array(ac)[post*i:post*(i+1)],axis=0) for i in range(n*d)]
        ac_std=[np.std(np.array(ac)[post*i:post*(i+1)],axis=0) for i in range(n*d)]
        print('\nCluster Similarity:')
        if std[1]:
            info=np.array([f'{ac_mean[i]:.2f}'+f' ({ac_std[i]:.2f})' for i in range(n*d)]).reshape((d,n))
        else:
            info=np.array([f'{ac_mean[i]:.2f}' for i in range(n*d)]).reshape((d,n))
        print(pd.DataFrame(data=info,columns=clus,index=dims))

        ari_mean=[np.mean(np.array(ari)[post*i:post*(i+1)],axis=0) for i in range(n*d)]
        ari_std=[np.std(np.array(ari)[post*i:post*(i+1)],axis=0) for i in range(n*d)]
        print('\nAdjusted Rand Index:')
        if std[1]:
            info=np.array([f'{ari_mean[i]:.2f}'+f' ({ari_std[i]:.2f})' for i in range(n*d)]).reshape((d,n))
        else:
            info=np.array([f'{ari_mean[i]:.2f}' for i in range(n*d)]).reshape((d,n))
        print(pd.DataFrame(data=info,columns=clus,index=dims))

def centrality(
    W,
    alpha=1,
    posfun=np.abs,
    pkg="igraph",
    all_shortest_paths=False,
    weighted=True,
    signed=True,
    R2=False
):
    """
    Python translation of R's qgraph::centrality()
    -------------------------------------------------
    W : numpy array
        Directed adjacency matrix (weights allowed)
    alpha : float
        Exponent for degree-weighted centrality
    posfun : function
        Function to transform weights (abs by default)
    pkg : str
        "igraph" or "qgraph" behavior (only "igraph" behavior implemented)
    all_shortest_paths : bool
        If True computes full shortest-path lists
    weighted : bool
        If False uses unweighted edges
    signed : bool
        If False uses absolute weights
    R2 : bool
        If True computes GGM-based R2 measure
    """

    # Copy matrix
    W = np.array(W, dtype=float)
    n = W.shape[0]

    # If not weighted -> binarize
    if not weighted:
        W = np.sign(W)

    # If not signed -> absolute value
    if not signed:
        W = np.abs(W)

    # Remove self-loops
    np.fill_diagonal(W, 0)

    ### --- BASIC ADJACENCY ---
    X = (W != 0).astype(int)

    ### --- OUT DEGREE ---
    unweighted_out = X.sum(axis=1)
    weighted_out = posfun(W).sum(axis=1)
    combined_out = (unweighted_out ** (1 - alpha)) * (weighted_out ** alpha)

    ### --- IN DEGREE ---
    unweighted_in = X.sum(axis=0)
    weighted_in = posfun(W).sum(axis=0)
    combined_in = (unweighted_in ** (1 - alpha)) * (weighted_in ** alpha)

    ### --- EXPECTED INFLUENCE ---
    out_expected = W.sum(axis=1)
    in_expected = W.sum(axis=0)

    ### ---------------------------------------------
    ### GRAPH SETUP FOR SHORTEST PATH METRICS
    ### ---------------------------------------------
    # Distances = 1 / |w|^alpha
    D = np.zeros_like(W)
    mask = posfun(W) != 0
    D[mask] = 1.0 / (posfun(W)[mask] ** alpha)
    D[~mask] = np.inf

    # Build directed weighted graph
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if np.isfinite(D[i, j]) and D[i, j] != 0:
                G.add_edge(i, j, weight=D[i, j])

    # --- Closeness (directed)
    closeness = np.array([nx.closeness_centrality(G, u=i, distance="weight") for i in range(n)])

    # --- Betweenness
    betw = np.array(list(nx.betweenness_centrality(G, weight="weight", normalized=False).values()))

    # --- Shortest path length matrix ---
    shortest_matrix = np.full((n, n), np.inf)
    for i in range(n):
        lengths = nx.single_source_dijkstra_path_length(G, i, weight="weight")
        for j, d in lengths.items():
            shortest_matrix[i, j] = d

    # --- List all shortest paths ---
    shortest_paths = [[[] for _ in range(n)] for __ in range(n)]
    if all_shortest_paths:
        for i in range(n):
            for j in range(n):
                if np.isfinite(shortest_matrix[i, j]) and i != j:
                    shortest_paths[i][j] = list(nx.all_shortest_paths(G, source=i, target=j, weight="weight"))

    ### ------------------------------------------------
    ### R2 METRIC (GGM-like) -- optional
    ### ------------------------------------------------
    R2_res = None
    if R2:
        K = np.eye(n) - W
        if not (np.allclose(K, K.T) and np.all(np.linalg.eigvals(K) > 0)):
            raise ValueError("Matrix does not appear to be a valid GGM precision matrix.")

        # convert to standardized covariance inverse
        K_inv = np.linalg.inv(K)
        K_std = np.linalg.inv((1 / np.sqrt(np.diag(K_inv)))[:, None] * K_inv * (1 / np.sqrt(np.diag(K_inv)))[None, :])
        R2_res = 1 - 1 / np.diag(K_std)

    ### --- Assemble output ---
    results = {
        "OutDegree": combined_out,
        "InDegree": combined_in,
        "Closeness": closeness,
        "Betweenness": betw,
        "InExpectedInfluence": in_expected,
        "OutExpectedInfluence": out_expected,
        "ShortestPathLengths": shortest_matrix,
        "ShortestPaths": shortest_paths if all_shortest_paths else None
    }

    if R2:
        results["R2"] = R2_res

    return results

def adjacency_matrix(result):
    """helper function that computes the adjacency matrix to be used in network visualization from vista algorithm parameters

    Parameters
    ----------
    result: dictionary
        output from vista algorithm

    Returns
    -------
    list
        list of adjacency matrices for each cluster
    """
    mtx=[]
    for i in range(len(result['parameter']['C'])):
        A=(result['parameter']['C'][i])@(result['parameter']['A'][i])@np.linalg.pinv(result['parameter']['C'][i])
        mtx.append(A)
    return mtx

def temporal_network(result,features,cluster_labels,seed=1,dpi=300,filename=''):
    """plots temporal network visualization given the result of vista algorithm

    Parameters
    ----------
    result: dictionary
        output from vista algorithm
    features: list
        labels for the nodes in the temporal network corresponding to the features in the observed data
    cluster_labels: list
        labels for the clusters given by the vista output
    seed: int, optional
        random seed for graph layout
    dpi: int, optional
        plot resolution
    filename: string, optional
        filename to which, if provided, the plot will be saved
    """
    mtx=adjacency_matrix(result)
    layouts = [nx.spring_layout(nx.from_numpy_array(matrix, create_using=nx.DiGraph), seed=seed) for matrix in mtx]
    n=len(features)
    m=len(mtx)
    avg_layout = {
        node: sum([layout[node] for layout in layouts])/m
        for node in range(n)
    }

    fig, axes = plt.subplots(1, m, figsize=(14, 7))
    
    for ax, (matrix, title) in zip(axes,[(mtx[i],cluster_labels[i]) for i in range(m)]):
        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        edges = G.edges()
        weights = [matrix[i, j] for i, j in edges]
        colors = ["#0080FF" if w >= 0 else "#f73a29" for w in weights]
        wm=max([abs(w) for w in weights])
        widths = [abs(w)/wm * 5 for w in weights]
    
        nx.draw_networkx_nodes(G, pos=avg_layout, node_color="#d9d9d9", node_size=2000, ax=ax)
        nx.draw_networkx_labels(G, pos=avg_layout,
                                labels={i: features[i] for i in range(n)},
                                font_size=14, font_weight='bold', ax=ax)
        nx.draw_networkx_edges(
            G, pos=avg_layout,
            edge_color=colors,
            width=widths,
            arrows=True,
            arrowstyle='-|>', arrowsize=15,
            connectionstyle='arc3,rad=0.15',
            ax=ax,
            node_size=2000
        )
    
        ax.set_title(title, fontsize=16)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()
    if len(filename)>0:
        plt.savefig(filename, dpi=dpi)

def out_expected_influence(result,features,cluster_labels,dpi=300,filename=''):
    """plots out-expected influence given the result of vista algorithm, using qgraph's centrality function

    Parameters
    ----------
    result: dictionary
        output from vista algorithm
    features: list
        labels for the nodes in the temporal network corresponding to the features in the observed data
    cluster_labels: list
        labels for the clusters given by the vista output
    dpi: int, optional
        plot resolution
    filename: string, optional
        filename to which, if provided, the plot will be saved
    """
    mtx=adjacency_matrix(result)
    influence_dic={"label": features}
    for i in range(len(cluster_labels)):
        influence_dic[cluster_labels[i]]=centrality(mtx[i])['OutExpectedInfluence']
    df = pd.DataFrame(influence_dic)
    y_positions = np.arange(len(features))
    
    plt.figure(figsize=(7, 9))
    plt.axvline(0, color="gray", linestyle="dashed", linewidth=1)

    for i, label in enumerate(features):
        vals=[df.loc[df["label"] == label, cl].values[0] for cl in cluster_labels]
        plt.plot([min(vals), max(vals)], [i, i], color="lightgray", linewidth=2, zorder=1)

    for cl in cluster_labels:
        plt.scatter(df[cl], y_positions, s=80, label=cl, zorder=2)

    plt.yticks(y_positions, features)
    plt.xlabel("Out-Expected Influence", fontsize=12)
    plt.ylabel("")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=len(cluster_labels), frameon=False)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if len(filename)>0:
        plt.savefig(filename, dpi=dpi)
    plt.show()
