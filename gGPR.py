#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular, cholesky
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class gGPR:
    def __init__(self, x_train, y_train, alpha=np.array([1.e-10]), cbar2=1, ell=1):
        """
        RBF kernel is assumed
        Args:
            x_train: Training input data (numpy array).
            y_train: Training target data (numpy array).
            alpha: diagonal of covariance (numpy array).

        Kernel = c^2 exp( x-x' / 2l^2 )
        """
        self.x_train = x_train
        self.y_train = y_train
        if(alpha[0]==1.e-10): self.sigma = np.diag(np.array([1.e-10]*np.size(y_train)))
        else: self.sigma = np.diag(alpha)
        k1 = ConstantKernel(constant_value=cbar2)
        k2 = RBF(length_scale=ell)
        kernel = k1 * k2
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=alpha)
        self.gpr.fit(x_train.reshape(-1,1), y_train)

    def get_kernel_scale(self):
        """
        return c^2
        """
        return self.gpr.kernel_.k1.constant_value
    def get_length_scale(self):
        """
        return l
        """
        return self.gpr.kernel_.k2.length_scale

    def predict(self, x, k=0, theta=[None]):
        """
        Args:
            x: Input data for the prediction (numpy array).
            k: order of derivative (integer, k >= 0)

        Returns:
            m: Mean of the posterior GP at the x points (numpy array).
            cov: Covariance of the posterior GP at the x points (numpy array).
        """
        from scipy.special import eval_hermite
        if(theta[0]==None): 
            theta = self.gpr.kernel_.theta
        kernel = self.gpr.kernel_.clone_with_theta(theta)
        #if(k==0): return self.gpr.predict(x.reshape(-1,1), return_cov=True)
        K = kernel(self.x_train.reshape(-1,1), self.x_train.reshape(-1,1)) + self.sigma
        L = cholesky(K, lower=True)
        Kinv = np.linalg.inv(K)
        ell = np.exp(theta[1])
        fact = 1 / (np.sqrt(2)*ell)
        z = (x.reshape(-1,1) - self.x_train.reshape(1,-1)) * fact
        Z = (x.reshape(-1,1) - x.reshape(1,-1)) * fact
        dkappa_l = (-1)**k * fact**k * eval_hermite(k, z) * kernel(x.reshape(-1,1), self.x_train.reshape(-1,1)) # Drischler et al. Phys. Rev. C 102, 054315
        dkappa = (-1)**k * fact**(2*k) * eval_hermite(2*k, Z) * kernel(x.reshape(-1,1), x.reshape(-1,1)) # Drischler et al. Phys. Rev. C 102, 054315
        v = solve_triangular(L, (-1)**k * dkappa_l.T, lower=True, check_finite=False) 
        mu = dkappa_l @ Kinv @ self.y_train
        cov = dkappa - v.T @ v 
        return mu, cov

    def get_hyperparameter_dist(self, n_sample=1000, width=0.05):
        theta = self.gpr.kernel_.theta
        samples = []
        cnt = 0
        while(len(samples) < n_sample):
            theta_new = [np.random.normal(p, width) for p in theta]
            a = np.exp(self.gpr.log_marginal_likelihood(theta_new) - self.gpr.log_marginal_likelihood(theta))
            is_accept = np.random.uniform() < min(1, a)
            if(is_accept): 
                cnt += 1
                theta = theta_new
            samples.append(theta)
        print("accept rate: {:.6f}%".format(cnt/len(samples)*100))
        return np.exp(np.array(samples))

    def marginal_prediction(self, x_test, k=0, n_sample=100, width=0.1, params=[]):
        if(len(params)==0):
            res = self.get_hyperparameter_dist(n_sample=n_sample, width=width)
            params = pd.DataFrame({"c2":res[:,0], "l":res[:,1]})
        means, covs = np.zeros(x_test.shape), np.zeros((x_test.shape[0],x_test.shape[0]))
        for idx, row in params.iterrows():
            theta = np.log(row.to_numpy())
            m, cov = self.predict(x_test, k=k, theta=theta)
            means += m / len(params)
            covs += cov / len(params)
        return means, covs

if(__name__=="__main__"):
    from scipy.special import spherical_jn
    def true_func(x, k=0):
        #return spherical_jn(0, x, k)

        # sin(x)
        #if(k%4==0): return np.sin(x)
        #if(k%4==1): return np.cos(x)
        #if(k%4==2): return -np.sin(x)
        #if(k%4==3): return -np.cos(x)

        # 0th-order spherical bessel
        if(k==0): return spherical_jn(0,x)
        if(k==1): return -spherical_jn(1,x)
        if(k==2): return -spherical_jn(0,x) + 2/x * spherical_jn(1,x)
        if(k==3): return 2/x * spherical_jn(0,x) + (1-6/x**2) * spherical_jn(1,x)
            

    n_train = 4
    n_test = 100
    x_train = np.linspace(1.e-4, 4.e-2 * np.pi, n_train)
    x_train = np.append(x_train,np.array([0.2,0.3,0.5]))

    n_train = 8
    x_train = np.linspace(1.e-4, 4.e0 * np.pi, n_train)

    y_train = true_func(x_train) 
    x_test = np.linspace(1.e-4, 4.e-0 * np.pi, n_test)
    GP = gGPR(x_train, y_train)

    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(10)
    k=0
    m, cov = GP.predict(x_test, k=k)
    ax.scatter(x_train, y_train, label='Training Data', color='blue')
    ax.plot(x_test, true_func(x_test,k=k), label='f(x)', color='black', linestyle='--')
    ax.plot(x_test, m, label='posterior mean', color='r', alpha=0.5)
    ax.fill_between(x_test, m-1.96*np.sqrt(cov.diagonal()), m+1.96*np.sqrt(cov.diagonal()), alpha=0.2, color='r')

    k=1  
    m, cov = GP.predict(x_test, k=k) # first derivative
    ax.plot(x_test, true_func(x_test,k=k), color='black', linestyle='--')
    ax.plot(x_test, m, label='posterior mean (1st deriv.)', color='b', alpha=0.5)
    ax.fill_between(x_test, m-1.96*np.sqrt(cov.diagonal()), m+1.96*np.sqrt(cov.diagonal()), alpha=0.2, color='b')

    k=2
    m, cov = GP.predict(x_test, k=k) # second derivative
    ax.plot(x_test, true_func(x_test,k=k), color='black', linestyle='--')
    ax.plot(x_test, m, label='posterior mean (2nd deriv.)', color='g', alpha=0.5)
    ax.fill_between(x_test, m-1.96*np.sqrt(cov.diagonal()), m+1.96*np.sqrt(cov.diagonal()), alpha=0.2, color='g')

    k=3
    m, cov = GP.predict(x_test, k=k) # third derivative
    ax.plot(x_test, true_func(x_test,k=k), color='black', linestyle='--')
    ax.plot(x_test, m, label='posterior mean (3rd deriv.)', color='violet', alpha=0.5)
    ax.fill_between(x_test, m-1.96*np.sqrt(cov.diagonal()), m+1.96*np.sqrt(cov.diagonal()), alpha=0.2, color='violet')

    ax.legend()

    res = GP.get_hyperparameter_dist(n_sample=2000, width=0.05)
    df = pd.DataFrame({"c2":res[:,0], "l":res[:,1]})

    g = sns.PairGrid(df, corner=True, vars=["c2","l"])
    g.map_lower(sns.kdeplot, fill=True, alpha=0.5, levels=[0.01,0.05,0.32,1])
    g.map_lower(sns.scatterplot, alpha=0.2, s=5, color="g")
    g.map_diag(sns.histplot, kde=True)


    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(10)

    k = 0
    m, cov = GP.marginal_prediction(x_test, k=k, params=df)
    ax.scatter(x_train, y_train, label='Training Data', color='blue')
    ax.plot(x_test, true_func(x_test,k=k), label='f(x)', color='black', linestyle='--')
    ax.plot(x_test, m, label='posterior mean', color='r', alpha=0.5)
    ax.fill_between(x_test, m-1.96*np.sqrt(cov.diagonal()), m+1.96*np.sqrt(cov.diagonal()), alpha=0.2, color='r')

    k = 1
    m, cov = GP.marginal_prediction(x_test, k=k, params=df)
    ax.plot(x_test, true_func(x_test,k=k), color='black', linestyle='--')
    ax.plot(x_test, m, label='posterior mean (1st deriv.)', color='b', alpha=0.5)
    ax.fill_between(x_test, m-1.96*np.sqrt(cov.diagonal()), m+1.96*np.sqrt(cov.diagonal()), alpha=0.2, color='b')

    k = 2
    m, cov = GP.marginal_prediction(x_test, k=k, params=df)
    ax.plot(x_test, true_func(x_test,k=k), color='black', linestyle='--')
    ax.plot(x_test, m, label='posterior mean (2nd deriv.)', color='g', alpha=0.5)
    ax.fill_between(x_test, m-1.96*np.sqrt(cov.diagonal()), m+1.96*np.sqrt(cov.diagonal()), alpha=0.2, color='g')

    k = 3
    m, cov = GP.marginal_prediction(x_test, k=k, params=df)
    ax.plot(x_test, true_func(x_test,k=k), color='black', linestyle='--')
    ax.plot(x_test, m, label='posterior mean (3rd deriv.)', color='violet', alpha=0.5)
    ax.fill_between(x_test, m-1.96*np.sqrt(cov.diagonal()), m+1.96*np.sqrt(cov.diagonal()), alpha=0.2, color='violet')
    plt.show()