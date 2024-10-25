import multiprocessing as mp
import _thread as thread
import numpy as np

from .kalmanfilter_and_smoother import add_input_to_state, add_input_to_obs
from .em_algorithm_for_lgssm import EMlgssm
from .utils.pinv import pseudo_inverse

np.seterr(over='raise')
#np.seterr(all='warn')

#https://stackoverflow.com/questions/9068478/how-to-parallelize-a-sum-calculation-in-python-numpy
class Sum: 
    def __init__(self,dims):
        (dx,dy)=dims
        self.value=np.array([np.zeros((dx,1)),np.zeros((dx,dx)),np.zeros((dx,dx)),np.zeros((dx,dx))
             ,np.zeros((dx,dx)),np.zeros((dx,dx)),np.zeros((dx,dx)),np.zeros((dx,dx)),np.zeros((dx,dx)),0,0
             ,np.zeros((dy,dy)),np.zeros((dy,dx))],dtype=object)
        self.lock = thread.allocate_lock()
        self.count = 0

    def add(self,value):
        self.count += 1
        self.lock.acquire() #lock so sum is correct if two processes return at same time
        self.value = self.value + value #the actual summation
        self.lock.release()

class EMmlgssm(object):
    """
    EM algorithm for Mixtures of Linear Gaussian State Space Models (MLGSSM).

    Model:
        x_i[t] = A^(k)*x_i[t-1] (+ B^(k)*u_x[t]) + w_i[t]
        y_i[t] = C^(k)*x_i[t] (+ D^(k)*u_y[t]) + v_i[t]  
        x_i[1] = mu^(k) + u_i
        where
        w_i[t] ~ N(0, Gamma^(k))
        v_i[t] ~ N(0, Sigma^(k))
           u_i ~ N(0, P^(k))

    Parameters: 
        A^(k), Gamma^(k), C^(k), Sigma^(k), mu^(k), P^(k)(, B^(k), D^(k))
    """
    
    def __init__(self,
        state_mats, 
        state_covs, 
        obs_mats, 
        obs_covs, 
        init_state_means, 
        init_state_covs, 
        weights,
        input_state_mats=None, 
        input_obs_mats=None
        ):
        """
        Set parameters.

        Arguments
        ---------
        state_mats : np.ndarray(n_clusters, dim_x, dim_x)
            Set of A.
        state_covs : np.ndarray(n_clusters, dim_x, dim_x)
            Set of Gamma.
        obs_mats : np.ndarray(n_clusters, dim_y, dim_x)
            Set of C.
        obs_covs : np.ndarray(n_clusters, dim_y, dim_y)
            Set of Sigma.
        init_state_means : np.ndarray(n_clusters, dim_x, 1)
            Set of mu.
        init_state_covs : np.ndarray(n_clusters, dim_x, dim_x)
            Set of P.
        weights : np.ndarray(n_clusters, 1, 1)
            Set of pi.
        input_state_mats : np.ndarray(n_clusters, dim_x, dim_ux), default=None 
            Set of B.
        input_obs_mats : np.ndarray(n_clusters, dim_y, dim_uy), default=None
            Set of D.
        """

        self.set_A = state_mats
        self.set_Gamma = state_covs
        self.set_C = obs_mats
        self.set_Sigma = obs_covs
        self.set_mu = init_state_means
        self.set_P = init_state_covs
        self.set_pi = weights
        self.set_B = input_state_mats
        self.set_D = input_obs_mats

        self.M, self.d_y, self.d_x = self.set_C.shape

        if add_input_to_state(self.set_B):
            self.d_ux = self.set_B.shape[2]
        if add_input_to_obs(self.set_D):
            self.d_uy = self.set_D.shape[2]

        
    def _init_em_lgssm(self, data_num, cluster_num):
        """
        Create 'EMlgssm'-instance.

        Arguments
        ---------
        data_num : int
            Data number.

        cluster_num : int
            Cluster number.

        Returns
        -------
        lgssm : instance.
            Instance of class 'EMlgssm'.
        """

        lgssm = EMlgssm(
            state_mat=self.set_A[cluster_num], 
            state_cov=self.set_Gamma[cluster_num], 
            obs_mat=self.set_C[cluster_num], 
            obs_cov=self.set_Sigma[cluster_num],
            init_state_mean=self.set_mu[cluster_num], 
            init_state_cov=self.set_P[cluster_num],
            time_points = self.set_time[data_num],
            data_loc=(cluster_num,data_num)
        )

        lgssm.y = self.set_y[data_num]
        lgssm.u_x = self.u_x
        lgssm.u_y = self.u_y

        if add_input_to_state(self.set_B):
            lgssm.B = self.set_B[cluster_num]
        if add_input_to_obs(self.set_D):
            lgssm.D = self.set_D[cluster_num]

        return lgssm
    
    
    def _e_step(self, index, stage=False):
        """
        Base of E-step.

        Arguments
        ---------
        index : tuple
            (data_num, cluster_num).

        stage : bool
            (which part of e step to run)

        Returns
        -------
        Results of E-step for specific numbers i and k.
        """
        
        (i, k) = index
        
        # Create 'EMlgssm'-instance
        lgssm = self._init_em_lgssm(data_num=i, cluster_num=k)

        # Run E-step of EM algorithm for LGSSM
        lgssm.run_e_step(stage=stage)
        if stage==True:
            e_xt = lgssm.e_xt
            e_xtxt = lgssm.e_xtxt
            e_xtxt_1 = lgssm.e_xtxt_1
            return (e_xt, e_xtxt, e_xtxt_1)
        else:
            # Sum of lgssm-log-likelihood and log-weight
            loglikelihood = (
                lgssm.compute_likelihoods(
                    x_means_p=lgssm.means_p, x_covs_p=lgssm.covs_p, log=True
                ) 
                + 
                np.log(self.set_pi[k] + 1e-7)
            )
        
            return (i, k, loglikelihood)
    
    
    def _compute_posterior_prob(self, ll):
        """
        Compute posterior probabilities.

        Arguments
        ---------
        ll : np.ndarray(n_datas, n_clusters, 1, 1) 
            Log-likelihoods.

        Returns
        -------
        pp : np.ndarray(n_datas, n_clusters, 1, 1)
            Posterior probabilities.
        """

        ll[:,:,0] -= np.max(ll, axis=1)
        pp = np.einsum("nmij,njl->nmil", np.exp(ll), 1 / np.sum(np.exp(ll), axis=1))
        pp += 1e-7
        
        return pp

    def _update_weight(self, k):
        """
        Update pi^(k).

        Arguments
        ---------
        k : int
            Cluster number.
        
        Returns
        -------
        : np.ndarray(1, 1)
        """

        return np.mean(self.p_prob[:,k])
    
    def computation(self,index):
        (i, k) = index
        expectations = self._e_step(index,stage=True)
        e_xt = expectations[0]
        e_xtxt = expectations[1]
        e_xtxt_1 = expectations[2]
        time_points = self.set_time[i]
        diffs = time_points[1:]-time_points[:-1]
        dt = np.insert(diffs,0,diffs[0])
        T = len(time_points)
        prb = self.p_prob[i,k,0,0]
        
        calcs=[]
        calcs.append(e_xt[0]*prb)
        calcs.append(e_xtxt[0]*prb)
        calcs.append(np.sum(e_xtxt*dt[:,None,None],axis=0)*prb)
        calcs.append(np.sum(e_xtxt[:-1],axis=0)*prb)
        calcs.append(np.sum(e_xtxt[:-1]/dt[1:,None,None],axis=0)*prb)
        calcs.append(np.sum(e_xtxt[:-1]*dt[1:,None,None],axis=0)*prb)
        calcs.append(np.sum(e_xtxt[1:]/dt[1:,None,None],axis=0)*prb)
        calcs.append(np.sum(e_xtxt_1,axis=0)*prb)
        calcs.append(np.sum(e_xtxt_1/dt[1:,None,None],axis=0)*prb)
        calcs.append(np.array(prb*T))
        calcs.append(np.array(prb*(T-1)))
        calcs.append(np.sum(np.einsum("tij,tdj->tid", self.set_y[i], self.set_y[i])*dt[:,None,None],axis=0)*prb)
        calcs.append(np.sum(np.einsum("tij,tdj->tid", self.set_y[i], e_xt)*dt[:,None,None],axis=0)*prb)
        return np.array(calcs,dtype=object)

    def summers(self,num_iters,k):
        pool = mp.Pool(processes=1)

        sumArr = Sum((self.d_x,self.d_y)) #create an instance of callback class and zero the sum
        for index in range(num_iters):
            singlepoolresult = pool.apply_async(self.computation,((index,k),),callback=sumArr.add)

        pool.close()
        pool.join() #waits for all the processes to finish

        return sumArr.value

    def run_e_m_step(self):
        """
        Run E-step.
        """
        if len(self.set_y.shape)==1:
            self.N = self.set_y.shape[0]
        else:
            self.N, self.T = self.set_y.shape[:2]

        index_pairs = [(i, k) for i in range(self.N) for k in range(self.M)]

        results = []
        if self.cores==1:
            for index in index_pairs:
                results.append(self._e_step(index))
        elif self.cores>1:
            # Multiprocessing
            with mp.Pool(self.cores) as pool:
                async_result = pool.map_async(self._e_step, index_pairs)
                results = async_result.get()
        else:
            raise  ValueError('n_cpu must be positive integer.')
            
        results.sort()
        
        # Posterior probabilities
        self.p_prob = self._compute_posterior_prob(
            ll = np.asarray([row[2] for row in results]).reshape(self.N, self.M, 1, 1)
        )
        
        #log likelihood
        self.loglikelihoods = np.asarray(
            [row[2] for row in results]
        ).reshape(self.N, self.M)


        for k in range(self.M):
            # Update pi
            self.set_pi[k] = self._update_weight(k)

            # compute expectations and update parameters
            calc_sum=self.summers(self.N,k)            
            denom=np.sum(self.p_prob[:,k], axis=0)


            
            if not 'mu' in self.fix:
                # Update mu
                self.set_mu[k] = calc_sum[0]/denom
            if not 'P' in self.fix:
                # Update P
                self.set_P[k] = (calc_sum[1]-np.outer(self.set_mu[k],calc_sum[0])
                                 -np.outer(calc_sum[0],self.set_mu[k])+denom*np.dot(self.set_mu[k], self.set_mu[k].T))/denom
                self.set_P[k] = (self.set_P[k] + self.set_P[k].T)/2 
                #account for asymmetry introduced by numerical imprecision
            if not 'A' in self.fix:
                # Update A
                self.set_A[k] = np.dot((calc_sum[7]-calc_sum[3]), pseudo_inverse(calc_sum[5]))
            if not 'Gamma' in self.fix:
                # Update Gamma
                self.set_Gamma[k] = (calc_sum[6]-self.set_A[k]@(calc_sum[7].T)-calc_sum[8].T-calc_sum[7]@self.set_A[k].T
                                     -calc_sum[8]+calc_sum[4]+calc_sum[3]@self.set_A[k].T+self.set_A[k]@(calc_sum[3].T)
                                     +self.set_A[k]@calc_sum[5]@self.set_A[k].T)/calc_sum[10]
                self.set_Gamma[k] = (self.set_Gamma[k] + self.set_Gamma[k].T)/2 
                #account for asymmetry introduced by numerical imprecision
            if not 'C' in self.fix:
                # Update C
                self.set_C[k] = np.dot(calc_sum[12], pseudo_inverse(calc_sum[2]))
            if not 'Sigma' in self.fix:
                # Update Sigma
                self.set_Sigma[k] = (calc_sum[11]-self.set_C[k]@(calc_sum[12].T)-calc_sum[12]@self.set_C[k].T
                                     +self.set_C[k]@calc_sum[2]@self.set_C[k].T)/calc_sum[9]
                self.set_Sigma[k] = (self.set_Sigma[k] + self.set_Sigma[k].T)/2 
                #account for asymmetry introduced by numerical imprecision

        return self

    def compute_bic(self):
        """
        Compute BIC for MLGSSM.
        
        Returns
        ----------
        bic : float
            BIC for MLGSSM.
        """
        
        def logaddexp(ll):
            sm=ll[:,1]
            if ll.shape[1]>2:
                sm = logaddexp(ll[:,1:])
            return np.logaddexp(ll[:,0],sm)

        # Number of default lgssm parameters
        n_param = self.d_x*(1+3*self.d_x+self.d_y)+self.d_y**2
        # Number of optional lgssm parameters
        if add_input_to_state(self.set_B):
            n_param += 1
        if add_input_to_obs(self.set_D):
            n_param += 1
        # Number of fixed parameters
        
        if 'mu' in self.fix:
            n_param -= self.d_x
        if 'P' in self.fix:
            n_param -= self.d_x**2
        if 'A' in self.fix:
            n_param -= self.d_x**2
        if 'Gamma' in self.fix:
            n_param -= self.d_x**2
        if 'C' in self.fix:
            n_param -= self.d_y*self.d_x
        if 'Sigma' in self.fix:
            n_param -= self.d_y**2
        
        # Params
        k = (self.M*n_param + self.M - 1.)
        ll = np.sum(logaddexp(self.loglikelihoods))
        
        #information criterions
        bic = k * np.log(self.N) - 2*ll
        abic = k * np.log((self.N+2)/24) - 2*ll #sample-size adjusted BIC (SABIC, Sclove, 1987) 
        aic = 2*k - 2*ll
        aicc = aic + 2*k*(k+1)/(self.N-k-1)

        return (bic,abic,aic,aicc)


    def fit(self, Y, time_points, ux=None, uy=None, max_iter=10, epsilon=0.01, n_cpu=1, fix_param=[], bic=False):
        """
        Run EM algorithm.

        Arguments
        ---------
        Y : np.ndarray(n_datas, len_y, dim_y, 1)
            Time series dataset.
        time_points : np.ndarray(n_datas, len_y)
            Times of each observation in Y.
        ux : np.ndarray(len_y, dim_ux, 1), default=None
            Input time series u_x.
        uy : np.ndarray(len_y, dim_uy, 1), default=None
            Input time series u_y.
        max_iter : int, default=10
            Maximum iteration number.
        epsilon : float, default=0.01
            Threshold for convergence judgment.
        n_cpu : int, default=1
            Number of CPUs.
        fix_param : list, default=[]
            If you want to fix some parameters of LGSSM, you should
            add the corresponding names(str) to list
                'mu' -> mu,    'P' -> P      
                'A' -> A,  'Gamma' -> Gamma,  'B' -> B
                'C' -> C,  'Sigma' -> Sigma,  'D' -> D
            For example, "fix_param=['C']" means that observation matrix 
            is fixed (not updated).
        bic : bool, default=False
            If True, compute bic.
        
        Returns
        -------
        results : dict
        """

        self.set_y = Y
        self.set_time = time_points
        self.u_x = ux
        self.u_y = uy
        self.cores = n_cpu
        self.fix = fix_param

        like=[]
        like_total=[]
        tol=[]

        for i in range(max_iter):
            
            # Keep current parameters
            set_A = self.set_A.copy()
            set_Gamma = self.set_Gamma.copy()
            set_C = self.set_C.copy() 
            set_Sigma = self.set_Sigma.copy()
            set_mu = self.set_mu.copy()
            set_P = self.set_P.copy()
            if add_input_to_state(self.set_B):
                set_B = self.set_B.copy()
            if add_input_to_obs(self.set_D):
                set_D = self.set_D.copy()

            # Run E-step & M-step
            self.run_e_m_step()

            # Convergence judgment
            _diff = 0
            if add_input_to_state(self.set_B):
                _diff += np.abs((self.set_B - set_B).sum())
            if add_input_to_obs(self.set_D):
                _diff += np.abs((self.set_D - set_D).sum())
            diff = _diff + np.sum(
                np.abs(
                    [
                        (self.set_mu - set_mu).sum(),
                        (self.set_P - set_P).sum(),
                        (self.set_A - set_A).sum(),
                        (self.set_Gamma - set_Gamma).sum(),
                        (self.set_C - set_C).sum(),
                        (self.set_Sigma - set_Sigma).sum()
                    ]
                )
            )

            tol.append(diff)
            like.append(self.loglikelihoods.sum(axis=0))
            like_total.append(self.loglikelihoods)

            
            if diff < epsilon:
                print('Termination tolerance achieved in '+str(i)+' iterations.')
                break
        if (i==(max_iter-1)):
            print('Maximum iterations reached.')

        params = {
            'weight':self.set_pi, 
            'mu':self.set_mu, 
            'P':self.set_P,
            'A':self.set_A, 
            'Gamma':self.set_Gamma,
            'C':self.set_C, 
            'Sigma':self.set_Sigma
        }

        if add_input_to_state(self.set_B):
            params['B'] = self.set_B
        if add_input_to_obs(self.set_D):
            params['D'] = self.set_D

        # Clustering
        labels = np.argmax(self.p_prob, axis=1)
        labels=labels.reshape(-1)

        results = {
            'parameter':params,
            'label':labels,
            'tolerance':tol,
            'likelihood':like,
            'total_likelihood':like_total,
            'iterations':i
        }

        # Compute BIC
        if bic:
            results['bic'] = self.compute_bic()
        
        return results