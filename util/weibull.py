import numpy as np
import scipy
import scipy.optimize
import scipy.special
import functools
import math

class Weibull:
    # https://en.wikipedia.org/wiki/Weibull_distribution
    
    def __init__(self, lambd: float, beta: float): 
        """
        lambd: scale parameter in (0, infinity)
        beta: shape parameter in (0, infinity)
        """
        self.lambd = lambd
        self.beta = beta 

    
    def __repr__(self):  
        return " Weibull distr for lambda=% s, beta=% s \n " % (self.lambd, self.beta)
    

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        The probability density function of the Weibull distribution.
        """
        # only consider values > 0 and set rest to zero
        result = np.zeros_like(X).astype(float)

        non_negative_indices = X > 0
        non_negatives = X[non_negative_indices]
        result[non_negative_indices] = self.beta / self.lambd * (non_negatives / self.lambd) ** (self.beta - 1) * np.exp(- (non_negatives / self.lambd) ** self.beta)
        return result

    def cdf(self, X: np.ndarray) -> np.ndarray:
        """
        The cummulative probability density function of the Weibull distribution.
        """
        # only consider values > 0 and set rest to zero
        result = np.zeros_like(X).astype(float)
        non_negative_indices = X > 0
        non_negatives = X[non_negative_indices]
        result[non_negative_indices] = 1 - np.exp(- (non_negatives / self.lambd) ** self.beta)
        return result
    
    def n_raw_moment(self, n=1):
        # https://proofwiki.org/wiki/Raw_Moment_of_Weibull_Distribution
        return self.lambd ** n * scipy.special.gamma(1 + n / self.beta)
        
    @functools.cached_property
    def mode(self):
        return 0 if self.k <= 1 else self.lambd * ((self.k - 1) / self.beta) ** (1 / self.beta)

    @functools.cached_property
    def median(self):
        return self.lambd * scipy.special.gamma(1 + 1 / self.beta)
    
    def ml_lambda(X: np.ndarray, beta: float) -> float:
        """
        Compute the scale parameter lambda using the maximum likelihood method given a fixed beta.
        """
        assert len(X[X > 0]) > 0, "invalid input"
        N = X.shape[0]
        return (1 / N * np.sum(X ** beta)) ** (1 / beta)

    def ml_beta(X: np.ndarray) -> float:
        """
        Compute the shape parameter beta using the maximum likelihood method.
        """
        assert len(X[X > 0]) > 0, "invalid input"
        N = X.shape[0]
        l_fn = lambda beta: - 1 / N * np.sum(np.log(X)) - 1 / beta + np.sum(X ** beta * np.log(X)) / np.sum(X ** beta)
        return scipy.optimize.root(l_fn, 2.0)

    def estimate(X: np.ndarray):
        """
        Estimate the parameters of the Weibull distribution using the Maximum Likelihood Method.
        """
        # only consider positive values for the ML-estimation (log is only defined for positive numbers)
        X = X[X > 0]
        X= X[~np.isnan(X)]

        #assert len(X) > 0, "invalid input"
        # exception if no data is available for computation
        
        try:
            b = Weibull.ml_beta(X).x.item()
            l = Weibull.ml_lambda(X, b).item()
        except Exception:
            b=-999
            l=-999
        finally:
        
            return Weibull(l, b)
       
    
    
    def graphical_parameters(X: np.ndarray): 
        '''
        Compute the parameters of the weibull distribution with the graphical method 
        '''
        max_included_windspeed=int(np.nanmax(X)+1)
        number_of_bins=1000
        edges=np.linspace(0,max_included_windspeed, number_of_bins)
        #Start by computing the empirical CDF:
        empiric_pdf= np.histogram(X, bins=edges)[0]
        CDF=np.array([empiric_pdf[0]])
        for i in range(1, len(empiric_pdf)):
            CDF=np.append(CDF, CDF[i-1]+empiric_pdf[i] )
        
        # disregard the first d  bins? -> This makes sure we have no near-constant part in the beginning
        d=0
        for i in range(0, len(CDF)):
            d=i
            if CDF[i] >0.1:
                break

        edges=edges[d:]
        empiric_pdf=empiric_pdf[d:]
        CDF=CDF[d:]

        #normalize the CDF and the PDF
        empiric_pdf=(empiric_pdf/CDF[len(CDF)-1])
        CDF=CDF/CDF[len(CDF)-1]

        # transform the axes of the CDF in order to find the Weibull parameters
        mod_CDF= np.log(-np.log(1-CDF[1:] +0.000001) +0.000001) # HOW TO RESOLVE THIS PROPERLY?
        log_edges=np.log(edges[2:])

         # now do linear regression
        linpol=np.polyfit(log_edges, mod_CDF, 1)
        mod_CDF_model = np.poly1d(linpol)
        b=linpol[0]
        l=np.exp( - linpol[1]/linpol[0])
        return [l,b]
    
    

    def graphical_estimate(X: np.ndarray):
        """
        Estimate the parameters of the Weibull distribution using the Graphical Method.
        """
        # only consider positive values for the ML-estimation (log is only defined for positive numbers)
        X= X[X > 0]
        X= X[~np.isnan(X)]

        #assert len(X) > 0, "invalid input"
        # exception if no data is available for computation
        try:
            params=Weibull.graphical_parameters(X)
        except Exception: 
            params=[-999, -999]
        finally:
            l=params[0]
            b=params[1]
            return Weibull(l,b)
        
    def epf_estimate(X: np.ndarray):
        """
        Estimate the parameters of the Weibull distribution using the Energy Pattern Factor Method
        """
        # only consider positive values for the ML-estimation (log is only defined for positive numbers)
        X= X[X > 0]
        X= X[~np.isnan(X)]

        # exception if no data is available for computation
        try:
            enum=0
            for v in X:
                enum=enum +v**3
            
            enum= enum/len(X)

            denom=0
            for v in X:
                denom=denom +v
            
            denom=denom/len(X)
            denom =denom**3
            
            epf=enum/denom
            b=1+ 3.69/(epf**2)
            l=X.mean()/scipy.special.gamma(1 + 1 /b)
        except Exception: 
            l=-999
            b=-999
        finally:
            return Weibull(l,b)
    

    def rel_fit(self, X: np.array, k: int) -> float:
        '''
        Computes the relative fit of the Weibull distribution to X, 
        i.e. for all the empiric data of X is sorted in k bins and the 
        expected relative error for a bin is calculated
        '''
        err=0.0
        max_wind=int(np.nanmax(X)+1)
        
        edges=np.linspace(0,max_wind, k+1)
        empiric_pdf= np.histogram(X, bins=edges)[0]
        empiric_pdf=empiric_pdf/(empiric_pdf.sum())
        for i in range(0, len(edges)-1):
            if empiric_pdf[i] >0:
                err=err + empiric_pdf[i]*abs(empiric_pdf[i] - (self.cdf(edges[i+1]) -self.cdf(edges[i])))/empiric_pdf[i] 
                
        return err

    def fit(self, X: np.array):
        '''
        Computes metrics of goodness of fit of the weibull class self to the data presented in the frame X
        The metrics are (as ordered in the output: MSE of the PDF, MSE of the CDF, R^2 of the PDF, R^2 of th CDF, KL-Divergence)
        '''
        # find an appropriate number of bins to sort in, as suggested in the lecture
        X=X.dropna().to_numpy()
        n_bins=int(0.1*np.sqrt(len(X)))+ 2 #int(0.1* np.sqrt(len(X)))+2
        #print(f'number of bins for fitting is {n_bins}')
        max_included_windspeed=int(np.nanmax(X)+1)
        number_of_bins=n_bins
        edges=np.linspace(0,max_included_windspeed, number_of_bins)
        empiric_pdf= np.histogram(X, bins=edges)[0]
        empiric_pdf=empiric_pdf/(empiric_pdf.sum())

        # first do mse
        mse=0
        for i in range(0, len(edges)-1):
            mse=mse + (empiric_pdf[i] - (self.cdf(edges[i+1]) -self.cdf(edges[i])))**2
        mse=1/len(empiric_pdf)*mse

        # do mse for the cdf
        
        cdf_mse=0
        F_i=0
        for i in range(0, len(edges)-1):
            F_i=F_i+empiric_pdf[i]
            cdf_mse=cdf_mse + (F_i - self.cdf(edges[i+1]))**2
        cdf_mse=1/len(empiric_pdf)*cdf_mse

        # then do r^2
        enum=0
        for i in range(0, number_of_bins-1):
            enum=enum+(empiric_pdf[i] - (self.cdf(edges[i+1]) -self.cdf(edges[i])))**2

        denom=0
        for i in range(0, number_of_bins-1):
            denom=denom+(empiric_pdf[i] - 1/number_of_bins)**2   

        r2=1-enum/denom

        # then do r^2 for the cdf
        cdf_enum=0
        F_i=0
        for i in range(0, number_of_bins-1):
            F_i=F_i+empiric_pdf[i]
            cdf_enum=cdf_enum+(F_i- self.cdf(edges[i+1]))**2

        mean_F_i= 1/number_of_bins*F_i
        F_i=0
        cdf_denom=0
        for i in range(0, number_of_bins-1):
            F_i=F_i+empiric_pdf[i]
            cdf_denom=cdf_denom+(F_i - mean_F_i)**2   

        cdf_r2=1-cdf_enum/cdf_denom
        
        # do the kl-divergence
        kl=0
        for i in range(0, len(edges)-1):
            kl=kl+empiric_pdf[i]*np.log(empiric_pdf[i]/ (self.cdf(edges[i+1]) -self.cdf(edges[i])+0.0001) +0.0001)


        return [mse.item(),cdf_mse.item(), r2.item(), cdf_r2.item(), kl]






