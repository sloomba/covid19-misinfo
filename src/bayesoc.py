class Dim():
    '''Model a univariate/1-D categorical/ordinal predictor variable'''
    
    def __init__(self, pi=1, ordinal=False, out=1, beta=0, delta=None, prior_beta=1., prior_delta=1., prior_mu_beta=1., prior_mu_delta=1., value=None, name='Dim'):
        import re
        self.name = '_'.join(re.split('[,|:|;|-|/|\s]+', ''.join(re.split("['|\.]+", str(name)))))
        self.set_pi(pi)
        self.set_ordinal(ordinal)
        self.set_out(out)
        self.set_beta(beta)
        self.set_delta(delta)
        self.set_prior_beta(prior_beta)
        self.set_prior_delta(prior_delta)
        self.set_prior_mu_beta(prior_mu_beta)
        self.set_prior_mu_delta(prior_mu_delta)
        self.set_value(value)

    def check_pi(self, x):
        import numpy as np
        if isinstance(x, int) or isinstance(x, float): x = np.ones(x)/x
        x = np.array(x)
        if (x>0).all() and np.allclose(x.sum(0), 1): return x
        else: raise ValueError('param for %s must be a non-negative vector that sums to 1'%self.name)
            
    def set_pi(self, pi=1):
        self.pi = self.check_pi(pi)
        self.num = len(self.pi)
        
    def __len__(self): return self.num
    
    def set_ordinal(self, ordinal=False):
        if ordinal and len(self)<=1: raise ValueError('can have an ordinal categorical only with more than 1 categories')
        self.ordinal = bool(ordinal)
        
    def set_out(self, out=1):
        assert(isinstance(out, int) and out>0)
        self.out = out
        
    def check_beta(self, x):
        import numpy as np
        if self.ordinal:
            if isinstance(x, float) or isinstance(x, int): x = x*np.ones(self.out)
            x = np.array(x)
            if len(x.shape)==1 and x.shape[0]==self.out: return x
            raise ValueError('expected a vector of param of size %i for %s'%(self.out, self.name))
        else:
            if isinstance(x, float) or isinstance(x, int): x = x*np.ones(len(self))
            x = np.array(x)
            if len(x.shape)==1: x = x[:,np.newaxis]*np.ones(self.out)[np.newaxis,:]
            if len(x.shape)==2 and x.shape[0]==len(self) and x.shape[1]==self.out: return x
            raise ValueError('expected a 2-d param matrix for %s of size (%i,%i)'%(self.name, len(self), self.out))
            
    def set_beta(self, beta=0): self.beta = self.check_beta(beta)    
            
    def check_delta(self, x):
        import numpy as np
        if x is None: x = 1/(len(self)-1)
        if isinstance(x, float) or isinstance(x, int): x = x*np.ones(len(self)-1)
        x = np.array(x)
        if len(x.shape)==1: x = x[:,np.newaxis]*np.ones(self.out)[np.newaxis,:]
        if len(x.shape)==2 and x.shape[0]==len(self)-1 and x.shape[1]==self.out: return x
        raise ValueError('expected a 2-d param matrix for %s of size (%i,%i)'%(self.name, len(self)-1, self.out))
        
    def set_delta(self, delta=None):
        if self.ordinal: self.delta = self.check_pi(self.check_delta(delta))
        else: self.delta = None
            
    def check_prior(self, x):
        assert(x>=0)
        return float(x)
            
    def set_prior_beta(self, prior_beta=1.): self.prior_beta = self.check_prior(prior_beta)
        
    def set_prior_delta(self, prior_delta=1.): self.prior_delta = self.check_prior(prior_delta)

    def set_prior_mu_beta(self, prior_mu_beta=1.): self.prior_mu_beta = self.check_prior(prior_mu_beta)
        
    def set_prior_mu_delta(self, prior_mu_delta=1.): self.prior_mu_delta = self.check_prior(prior_mu_delta)
        
    def set_value(self, value=None):
        if value is None: value = list(map(lambda x: str(x+1), range(len(self))))
        if len(value)!=len(self): ValueError('expected value for %s to be of size %i and not %i'%(self.name, len(self), len(value)))
        self.value = tuple(value)
        self.index = dict(zip(self.value, range(len(self))))
        
    def get_name(self): return [self.name]
            
    def get_beta(self, idx):
        '''get beta-contribution of given index'''
        if self.ordinal: return self.beta*(self.delta[:idx].sum(0))
        else: return self.beta[idx]

    def get_stan(self, outcome_size='m', outcome_index=':', hierarchical=True):
        dat = 'int<lower=1> k_%s;\nint<lower=1,upper=k_%s> %s[n];'%(self.name, self.name, self.name)
        if self.ordinal:            
            if self.out==1:
                par = 'real beta_%s;\nsimplex[k_%s-1] delta_%s;'%(self.name, self.name, self.name)
                mod = 'beta_%s ~ normal(0, %f);\n{\n\tvector[k_%s-1] u_%s;\n\tfor (i in 1:(k_%s-1))\n\t\tu_%s[i] = 1;\n\tdelta_%s ~ dirichlet(%f*u_%s);\n}'%(self.name, self.prior_beta, self.name, self.name, self.name, self.name, self.name, self.prior_delta, self.name)
                arg = 'beta_%s*sum(delta_%s[:%s[i]-1])'%(self.name, self.name, self.name)
            else:
                par = 'real beta_%s[%s];\nsimplex[k_%s-1] delta_%s[%s];'%(self.name, outcome_size, self.name, self.name, outcome_size)
                if hierarchical:
                    par += 'real mu_beta_%s; vector<lower=0>[k_%s] mu_delta_%s;'%(self.name, self.name, self.name)
                    mod = 'mu_beta_%s ~ normal(0, %f); mu_delta_%s ~ exponential(%f)'%(self.name, self.prior_mu_beta, self.name, self.prior_mu_delta)
                    mod += 'for (i in 1:%s)\n\tbeta_%s[i] ~ normal(mu_beta_%s, %f);\nfor (i in 1:%s)\n\tdelta_%s[i] ~ dirichlet(mu_delta_%s);'%(outcome_size, self.name, self.name, self.prior_beta, outcome_size, self.name, self.name)
                else: mod = 'for (i in 1:%s)\n\tbeta_%s[i] ~ normal(0, %f);\n{\n\tvector[k_%s-1] u_%s;\n\tfor (i in 1:(k_%s-1))\n\t\tu_%s[i] = 1;\n\tfor (i in 1:%s)\n\t\tdelta_%s[i] ~ dirichlet(%f*u_%s);\n}'%(outcome_size, self.name, self.prior_beta, self.name, self.name, self.name, self.name, outcome_size, self.name, self.prior_delta, self.name)
                arg = 'beta_%s*sum(delta_%s[%s,:%s[i]-1])'%(self.name, self.name, outcome_index, self.name)
        else:
            if self.out==1:
                par = 'real beta_%s[k_%s];'%(self.name, self.name)
                mod = 'for (i in 1:k_%s)\n\tbeta_%s[i] ~ normal(0, %f);'%(self.name, self.name, self.prior_beta)
                arg = 'beta_%s[%s[i]]'%(self.name, self.name)
            else:
                par = 'real beta_%s[%s,k_%s];'%(self.name, outcome_size, self.name)
                if hierarchical:
                    par += 'real mu_beta_%s[k_%s];'%(self.name, self.name)
                    mod = 'mu_beta_%s ~ normal(0, %f);'%(self.name, self.prior_mu_beta)
                    mod += 'for (i in 1:%s)\n\tfor (j in 1:k_%s)\n\t\tbeta_%s[i][j] ~ normal(mu_beta_%s[j], %f);'%(outcome_size, self.name, self.name, self.name, self.prior_beta)
                else: mod = 'for (i in 1:%s)\n\tfor (j in 1:k_%s)\n\t\tbeta_%s[i][j] ~ normal(0, %f);'%(outcome_size, self.name, self.name, self.prior_beta)
                arg = 'beta_%s[%s,%s[i]]'%(self.name, outcome_index, self.name)
        return {'data': dat, 'parameters': par, 'model': mod, 'output': arg}
    
    def __str__(self):
        stan = self.get_stan()
        return '\n\n'.join([k+'\n'+stan[k] for k in stan])
    
    def get_idx(self): return list(range(len(self)))
    
    def idx2val(self, idx): return self.value[idx]
    
    def val2idx(self, val): return self.index[val]
    
    def sample(self, size=1):
        from scipy.stats import multinomial
        return multinomial.rvs(n=1, p=self.pi, size=size).argmax(axis=1)
    
class Outcome():
    '''Model a univariate/1-D continuous/categorical/ordinal outcome variable'''
    
    def __init__(self, alpha=0, sigma=0.1, alpha_prior=1., sigma_prior=1., input=None, kind='ord', name='Outcome'):
        import re
        self.name = '_'.join(re.split('[,|:|;|-|/|\s]+', ''.join(re.split("['|\.]+", str(name)))))
        kinds = ['con', 'cat', 'ord']
        if kind not in kinds: raise ValueError('outcome type must be one of: "con"(tinuous), "cat"(egorical) or "ord"(inal)')
        self.kind = kind        
        self.set_alpha(alpha)
        self.set_sigma(sigma)
        self.set_alpha_prior(alpha_prior)
        self.set_sigma_prior(sigma_prior)
        self.set_input(input)
        
    def set_alpha(self, alpha=0):
        if self.kind=='con':
            assert(isinstance(alpha, int) or isinstance(alpha, float))
            self.alpha = float(alpha)
            self.num = 1
        elif self.kind=='cat':
            assert(isinstance(alpha, int) and alpha>0)
            self.alpha = None
            self.num = alpha
        elif self.kind=='ord':
            import numpy as np
            if isinstance(alpha, int) or isinstance(alpha, float): alpha = [alpha]
            alpha = np.array(alpha)
            if len(alpha.shape)==1 and alpha.shape[0]>=1:
                self.alpha = alpha
                self.num = 1+len(self.alpha)
            else: raise ValueError('expected alpha to be a vector of size greater than 0')
                
    def __len__(self): return self.num
                
    def set_sigma(self, sigma=0.1):
        if self.kind=='con':
            assert(sigma>0)
            self.sigma = float(sigma)
        else: self.sigma = None
                
    def check_prior(self, x):
        assert(x>=0)
        return float(x)
            
    def set_alpha_prior(self, alpha_prior=0.5): self.alpha_prior = self.check_prior(alpha_prior)
        
    def set_sigma_prior(self, sigma_prior=0.5): self.sigma_prior = self.check_prior(sigma_prior)
        
    def set_input(self, input=None):
        if input is None: #set default input
            if self.kind=='cat': self.input = Dim(out=len(self))
            else: self.input = None
        elif hasattr(input, 'get_beta'): self.input = input
        else: raise ValueError('expected input to be an object with "get_beta()" method')
            
    def get_name(self): return [self.name]
                
    def get_beta(self, idx):
        if self.input is None: return 0.
        else: return self.input.get_beta(idx)
        
    def get_stan(self, outcome_size='', hierarchical=True):
        if self.input is not None:
            if not outcome_size: outcome_size='k_%s'%self.name
            inp = self.input.get_stan(outcome_size=outcome_size, hierarchical=hierarchical)
            dat, par, mod, out = 'int<lower=1> n;\n'+inp['data']+'\n', inp['parameters']+'\n', inp['model']+'\n', inp['output']
            if self.kind=='con': out += ' + '
            elif self.kind=='cat': out = 'to_vector (%s) '%out
        else:
            dat, par, mod = 'int<lower=1> n;\n', '', ''
            if self.kind=='ord': out = '0.'
            else: out = ''
        if self.kind=='con':
            dat += 'real %s[n];'%self.name
            par += 'real alpha_%s;\nreal<lower=0> sigma_%s;'%(self.name, self.name)
            mod += 'alpha_%s ~ normal(0, %f);\nsigma_%s ~ exponential(%f);'%(self.name, self.alpha_prior, self.name, self.sigma_prior)
            mod += '\nfor (i in 1:n)\n\t%s[i] ~ normal (%salpha_%s, sigma_%s);'%(self.name, out, self.name, self.name)
        elif self.kind=='cat':
            dat += 'int<lower=1> k_%s;\nint<lower=1,upper=k_%s> %s[n];'%(self.name, self.name, self.name)
            mod += 'for (i in 1:n)\n\t%s[i] ~ categorical_logit (%s);'%(self.name, out)
        elif self.kind=='ord':
            dat += 'int<lower=1> k_%s;\nint<lower=1,upper=k_%s> %s[n];'%(self.name, self.name, self.name)
            par += 'ordered [k_%s-1] alpha_%s;'%(self.name, self.name)
            mod += 'for (i in 1:(k_%s-1))\n\talpha_%s[i] ~ normal(0, %f);'%(self.name, self.name, self.alpha_prior)
            mod += '\nfor (i in 1:n)\n\t%s[i] ~ ordered_logistic (%s, alpha_%s);'%(self.name, out, self.name)
        return {'data': dat, 'parameters': par, 'model': mod}
    
    def __str__(self):
        stan = self.get_stan()
        return '\n\n'.join([k+'\n'+str(stan[k]) for k in stan if stan[k]])
    
    def param(self, beta=0, alpha=None):
        if alpha is None: alpha = self.alpha
        if self.kind=='con': return alpha+beta
        elif self.kind=='cat':
            import numpy as np
            return np.exp(beta)/(np.exp(beta).sum())
        elif self.kind=='ord':
            import numpy as np
            return np.diff(np.hstack([0, np.exp(alpha-beta)/(1+np.exp(alpha-beta)), 1]))
    
    def sample(self, size=1):
        import numpy as np        
        if isinstance(size, int):
            if self.input is None: size = [0]*size
            else: size = self.input.sample(size)
        elif self.input is None: raise ValueError('no input model provided to index into')
        params = [self.param(self.get_beta(idx)) for idx in size]
        if self.kind=='con':
            from scipy.stats import norm
            return np.array([norm.rvs(loc=p, scale=self.sigma, size=1) for p in params])
        else:
            from scipy.stats import multinomial
            return np.array([multinomial.rvs(n=1, p=p, size=1).argmax() for p in params])
            
class Society():
    '''Model a multidimensional set of univariate/1-D predictor variables for univariate/1-D outcomes'''
    def __init__(self, ccy='£', name='Society'):
        import numpy as np
        self.name = str(name)
        self.currency = str(ccy)

        self.val = {'Sex': ['Male', 'Female'],
                    'Age': ['Under 16', '16-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', '86-95', 'Over 95'],
                    'Ethnicity': ['White', 'Black', 'Asian', 'Other'],
                    'Education': ['None', 'High School', 'Undergraduate', 'Postgraduate'],
                    'Employment': ['Employed', 'Unemployed', 'Student', 'Retired', 'Other'],
                    'Income': ['Under %s15k'%ccy, '%s15k-%s25k'%(ccy,ccy), '%s25k-%s35k'%(ccy,ccy), '%s35k-%s45k'%(ccy,ccy),
                               '%s45k-%s55k'%(ccy,ccy), '%s55k-%s65k'%(ccy,ccy), '%s65k-%s75k'%(ccy,ccy), 
                               '%s75k-%s85k'%(ccy,ccy), '%s85k-%s95k'%(ccy,ccy), 'Over %s95k'%ccy],
                    'Religion': ['Atheist', 'Catholic', 'Protestant', 'Jewish', 'Muslim', 'Hindu', 'Buddhist', 'Other'],
                    'Political': ['Center/Other', 'Left-leaning', 'Right-leaning']
                   }
        
        self.ord = {'Sex': False, 'Age': True, 'Ethnicity': False, 'Education': True, 'Employment': False, 'Income': True, 
                    'Religion': False, 'Political': False}

        self.par = {'Sex': (0, 0.2), # baseline Male
                    'Age': (2, (0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05)), #confidence increases with age
                    'Ethnicity': (0, -0.5, -0.2, 0.2), # baseline White
                    'Education': (3, (0.3, 0.5, 0.2)), # confidence increases with education
                    'Employment': (0, -1, -0.5, 2, 0), # baseline Employed
                    'Income': (-1, (0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.2, 0.2)), # confidence increases with income
                    'Religion': (0, -3, -2, -1, -3, -1, -1, 0), # baseline Atheist
                    'Political': (0, 2, -2), # baseline Other
                   }
        
        self.pis = {'Sex': [0.5, 0.5],
                    'Age': [0.2, 0.15, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.03, 0.02],
                    'Ethnicity': [0.8, 0.1, 0.05, 0.05],
                    'Education': [0.2, 0.4, 0.3, 0.1],
                    'Employment': [0.5, 0.1, 0.1, 0.2, 0.1],
                    'Income': [0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01],
                    'Religion': [0.25, 0.3, 0.3, 0.01, 0.05, 0.04, 0.01, 0.04],
                    'Political': [0.4, 0.3, 0.3]
                   }
        
        self.set_dims()        
    
    def set_dims(self):
        self.dims = []
        for d in self.val:
            if self.ord[d]: self.dims.append(Dim(pi=self.pis[d], ordinal=self.ord[d], beta=self.par[d][0], delta=self.par[d][1], value=self.val[d], name=d))
            else: self.dims.append(Dim(pi=self.pis[d], ordinal=self.ord[d], beta=self.par[d], value=self.val[d], name=d))
        self.num = len(self.dims)
        
    def __len__(self): return self.num
    
    def get_name(self):
        name = []
        for d in self.dims: name += d.get_name()
        return name
    
    def get_beta(self, idx): return sum([self.dims[i].get_beta(idx[i]) for i in range(len(self))])
    
    def get_stan(self):
        stan = {'data':[], 'parameters':[], 'model':[], 'output':[]}
        for i in range(len(self)):
            curr = self.dims[i].get_stan()
            for j in curr: stan[j].append(curr[j])
        for i in stan:
            if i=='output': stan[i] = ' + '.join(stan[i])
            else: stan[i] = '\n'.join(stan[i])
        return stan
    
    def __str__(self):
        stan = self.get_stan()
        return '\n\n'.join([k+'\n'+stan[k] for k in stan])
        
    def get_idx(self):
        from itertools import product
        return list(product(*[d.get_idx() for d in self.dims]))
    
    def idx2val(self, idx): return [self.dims[i].idx2val[idx[i]] for i in range(len(idx))]
    
    def val2idx(self, val): return [self.dims[i].val2idx[val[i]] for i in range(len(val))]
        
    def sample(self, size=1):
        '''Independently generate people across Blau dimensions'''
        import numpy as np
        return np.array([d.sample(size=size) for d in self.dims]).transpose()
        
class Model():
    '''Defines a class to model an outcome via Bayesian probabilistic programming (pystan)'''
    
    def __init__(self, outcome, name='Model'):
        self.name = str(name)
        self.outcome = outcome
        self.dat = None
        self.fit = None
        
    def param(self):
        import numpy as np
        return np.array([self.outcome.param(self.outcome.get_beta(idx)) for idx in self.outcome.input.get_idx()])
    
    def plot_param(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        params = self.param()
        if self.outcome.kind=='con':
            sns.distplot(params)
            plt.xlabel('$\\mu$')
        else:
            for i in range(params.shape[1]): sns.distplot(params[:,i], label='$i=%i$'%i)
            plt.xlabel('$P(Y=i)$')
            plt.legend()
        plt.show()
        
    def sample(self, size=1):
        import pandas as pd
        if self.outcome.input is not None:
            x = self.outcome.input.sample(size=size)
            df = pd.DataFrame(x, columns=self.outcome.input.get_name())
            df[self.outcome.name] = self.outcome.sample(size=x)
        else: df = pd.DataFrame(self.outcome.sample(size=size), columns=self.outcome.get_name())
        return df
    
    def get_stan(self):
        stan = self.outcome.get_stan()
        code = []
        for key in stan: code.append(key+' {\n\n'+stan[key]+'\n\n}')
        return '\n\n'.join(code)
    
    def __str__(self): return self.get_stan()
    
    def get_data(self, df):
        data_dict = {'n': df.shape[0]}
        if self.outcome.input is not None:
            for i in range(len(self.outcome.input)):
                try:
                    name = self.outcome.input.dims[i].name
                    data_dict['k_%s'%name] = len(self.outcome.input.dims[i])
                except:
                    name = self.outcome.input.name
                    data_dict['k_%s'%name] = len(self.outcome.input)
                data_dict['%s'%name] = 1+df[name].values #to do proper indexing
        name = self.outcome.name
        if self.outcome.kind=='con': data_dict['%s'%name] = df[name].values
        if self.outcome.kind in ['cat', 'ord']:
            data_dict['k_%s'%name] = len(self.outcome)
            data_dict['%s'%name] = 1+df[name].values
        return data_dict
    
    def run_stan(self, df=None, iter=2000):
        if df is None:
            if self.dat is None: raise ValueError('provide data sample(s) to run posterior inference on')
        else: self.dat = df.copy()
        import pystan as st
        code = self.get_stan()
        data = self.get_data(self.dat)
        model = st.StanModel(model_code=code)
        self.fit = model.sampling(data=data, iter=iter)        
        return self.fit
    
    def get_posterior_samples(self, pars=None, contrast=None, fit=None, as_df=True):
        if fit is None: fit = self.fit
        if pars is None: pars = fit.flatnames
        if contrast is not None and contrast not in pars: raise ValueError('variable to contrast with must be included in pars')
        samples = fit.extract(pars=pars, permuted=False)
        if contrast is not None: samples = {par+' - '+contrast: samples[par]-samples[contrast] for par in samples if par!=contrast}
        if as_df:
            import numpy as np
            import pandas as pd
            dat = []
            for par in samples: dat.append(samples[par])
            dat = np.dstack(dat)
            out = [[], []]
            for i in range(dat.shape[1]):
                out[0].append([i+1]*len(dat))
                out[1].append(dat[:,i,:])
            samples = pd.DataFrame(np.vstack(out[1]), columns=list(samples.keys()))
            samples['chain'] = np.hstack(out[0])
        return samples
        
    def get_posterior_params(self, idx=0, fit=None, as_df=True):
        import numpy as np
        if isinstance(idx, int) or isinstance(idx, float): idx = [idx]
        data = self.get_posterior_samples(fit=fit, as_df=True)
        if self.outcome.kind=='cat':            
            beta = []
            for i in range(len(self.outcome)): beta.append(np.vstack([data['beta_%s[%i,%i]'%(self.outcome.input.name, i+1, j+1)].values for j in idx]).sum(0))
            beta = np.vstack(beta).T
            data = np.array([self.outcome.param(beta=b) for b in beta]), data['chain'].values
        elif self.outcome.kind=='ord':
            alpha = np.vstack([data['alpha_%s[%i]'%(self.outcome.name, i+1)].values for i in range(len(self.outcome)-1)]).T
            if self.outcome.input is not None:
                if self.outcome.input.ordinal:
                    try: beta = data['beta_%s'%self.outcome.input.name].values*np.vstack([np.vstack([data['delta_%s[%i]'%(self.outcome.input.name, j+1)].values for j in range(i)]).sum(0) for i in idx]).sum(0)
                    except: beta = [0.]*len(alpha)
                else: beta = np.vstack([data['beta_%s[%i]'%(self.outcome.input.name, i+1)].values for i in idx]).sum(0)
            else: beta = [0.]*len(alpha)
            data = np.array([self.outcome.param(beta=b, alpha=a) for a, b in zip(alpha, beta)]), data['chain'].values
        elif self.outcome.kind=='con':
            alpha = data['alpha_%s'%self.outcome.name].values
            if self.outcome.input is not None: beta = np.vstack([data['beta_%s[%i]'%(self.outcome.input.name, i+1)].values for i in idx]).sum(0)
            else: beta = [0.]*len(alpha)
            data = np.array([self.outcome.param(beta=b, alpha=a) for a, b in zip(alpha, beta)]), data['chain'].values
        if as_df:
            import pandas as pd
            if self.outcome.kind=='con': out = pd.DataFrame(data[0], columns=['mu'])
            else: out = pd.DataFrame(data[0], columns=['p[%i]'%(i+1) for i in range(data[0].shape[1])])
            out['chain'] = data[1]
        else: 
            if self.outcome.kind=='con': out = {'mu': data[0]}
            else: out = {'p[%i]'%(i+1): data[0][:,i] for i in range(data[0].shape[1])}
        return out
    
    def get_posterior_stats(self, level=0.95, pars=None, contrast=None, params=False, idx=0, fit=None, as_df=True, plot=False):
        if params: samples = self.get_posterior_params(idx=idx, fit=fit, as_df=False)
        else: samples = self.get_posterior_samples(pars=pars, contrast=contrast, fit=fit, as_df=False)
        assert(0<=level<=1)
        low, med, upp = 0.5-level/2, 0.5, 0.5+level/2
        low_lab, med_lab, upp_lab = '%.1f%%'%(low*100), '%i%%'%(med*100), '%.1f%%'%(upp*100)
        stats = dict()
        for par in samples:
            all_samples = samples[par].flatten()
            all_samples.sort()
            stats[par] = {'Mean':all_samples.mean(), 'SD':all_samples.std(), low_lab:all_samples[int(low*len(all_samples))],
                          med_lab:all_samples[int(med*len(all_samples))], upp_lab:all_samples[int(upp*len(all_samples))]}
            if plot:
                import seaborn as sns
                import matplotlib.pyplot as plt
                sns.distplot(all_samples)
                plt.xlabel(par)
                plt.gca().axvline(stats[par]['Mean'], color='black', label='Mean')
                plt.gca().axvline(stats[par][low_lab], ls=':', label=low_lab)
                plt.gca().axvline(stats[par][med_lab], ls=':', label=med_lab)
                plt.gca().axvline(stats[par][upp_lab], ls=':', label=upp_lab)
                plt.legend()
                plt.show()
        if as_df:
            import pandas as pd
            stats = pd.DataFrame(stats).T
        return stats
    
    def plot_posterior_pairs(self, pars=None, contrast=None, fit=None):
        samples = self.get_posterior_samples(pars=pars, contrast=contrast, fit=fit, as_df=True)
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.pairplot(samples, hue='chain')
        plt.show()