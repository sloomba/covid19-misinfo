NUM_SAMPLES = 2000 #number of samples per chain; 4 chains total

def model_impact(df, group=1., kind='self', prior_mu=1., prior_sigma=1., iters=NUM_SAMPLES):
    # Model: Ref 1, Table 3
    # Results: Table 2
    import pystan as st
    model_code = '''
                    data {
                        int<lower=0> n; //number of data points
                        int<lower=1> m; //number of conditions
                        int<lower=2> k; //number of outcomes
                        int<lower=1,upper=k> y[n,m]; //outcome per sample per condition
                    }
                    parameters {
                        real mu[k-1];
                        real<lower=0> sigma[k-1];
                        ordered [k-1] alpha[m];
                    }
                    model {
                        for (i in 1:(k-1)) {
                            mu[i] ~ normal(0, %f);
                            sigma[i] ~ exponential(%f);
                        }
                        for (i in 1:m)
                            alpha[i] ~ normal(mu, sigma);
                        for (i in 1:n)
                            for (j in 1:m)
                                y[i,j] ~ ordered_logistic(0, alpha[j]);
                    }
                '''%(prior_mu, prior_sigma)
    df = df.loc[df['Treatment']==group]
    data = {'n':df.shape[0], 'm':2, 'k':4, 'y':df[['Vaccine Intent for %s (Pre)'%kind, 'Vaccine Intent for %s (Post)'%kind]].values}
    model = st.StanModel(model_code=model_code)
    fit = model.sampling(data=data, iter=iters)
    return fit

def model_impact_causal(df, kind='self', prior_mu=1., prior_sigma=1., prior_rho=1., iters=NUM_SAMPLES):
    # Model: Appendix C
    # Results: Tables S1, S2; Figures S1, S2
    import pystan as st
    model_code = '''
                data {
                    int<lower=0> n; //number of data points
                    int<lower=1> m; //number of conditions
                    int<lower=2> k; //number of outcomes
                    int<lower=1,upper=m> x_cond[n]; //treatment group
                    int<lower=1,upper=k> y_pre[n]; //pre-exposure outcome
                    int<lower=1,upper=k> y_post[n]; //post-exposure outcome
                }
                parameters {
                    real mu_alpha[k-1];
                    real mu_beta;
                    real<lower=0> sigma_alpha[k-1];
                    real<lower=0> sigma_beta;
                    real<lower=0> sigma_delta;
                    simplex[k-1] delta_delta;
                    simplex[k-1] delta[m];
                    real beta[m];
                    ordered[k-1] alpha[m];
                }
                model {
                    mu_alpha ~ normal(0, %f);
                    mu_beta ~ normal(0, %f);
                    sigma_alpha ~ exponential(%f);
                    sigma_beta ~ exponential(%f);
                    sigma_delta ~ exponential(%f);
                    {
                        vector[k-1] u;
                        for (i in 1:(k-1))
                            u[i] = 1;
                        delta_delta ~ dirichlet(%f*u);
                    }
                    for (i in 1:m){
                        beta[i] ~ normal(mu_beta, sigma_beta);
                        alpha[i] ~ normal(mu_alpha, sigma_alpha);
                        delta[i] ~ dirichlet(sigma_delta*delta_delta);
                    }
                    for (i in 1:n)
                        y_post[i] ~ ordered_logistic(beta[x_cond[i]]*sum(delta[x_cond[i]][:y_pre[i]-1]), alpha[x_cond[i]]);
                }
            '''%(prior_mu, prior_mu, prior_sigma, prior_sigma, prior_sigma, prior_rho)
    
    data = {'n':df.shape[0], 'm':2, 'k':4, 'x_cond':df['Treatment'].values+1, 
            'y_pre':df['Vaccine Intent for %s (Pre)'%kind].values, 
            'y_post':df['Vaccine Intent for %s (Post)'%kind].values}
    model = st.StanModel(model_code=model_code)
    fit = model.sampling(data=data, iter=iters)
    return fit

def model_socdem(df, dd, atts=[], group=None, kind='self', prior_beta=1., prior_delta=1., prior_alpha=1., iters=NUM_SAMPLES):
    # Model: Ref 2, 3, 4, 5 in Table 3
    # Results: Tables S3, S4, S5; Figures 3, 4
    import pystan as st
    import numpy as np
    from src.bayesoc import Dim #we define some helper classes to extract posterior samples easily
    cats = ['Age', 'Gender', 'Education', 'Employment', 'Religion', 'Political', 'Ethnicity', 'Income']
    if isinstance(atts, str): atts = [atts]
    for att in atts: cats += [x for x in list(df) if x[:len(att)]==att]
    outs = ['Vaccine Intent for self (Pre)', 'Vaccine Intent for self (Post)', 'Treatment']
    df = df[cats+outs].dropna()
    dims = [Dim(pi=len(dd[cat]), beta_prior=prior_beta, value=dd[cat].keys(), name=cat) for cat in cats]
    stan = [d.get_stan() for d in dims]
    code = {'data':[], 'parameters':[], 'model':[], 'output':[]}
    for key in code:
        for d in stan: code[key].append(d[key])        
    mod_cd = '''
                data {
                    int<lower=1> n; //number of data points
                    int<lower=2> k; //number of outcomes
                    int<lower=1,upper=k> y_pre[n]; //pre-exposure
                    int<lower=1,upper=k> y_post[n]; //post-exposure
                    %s
                }
                parameters {
                    %s
                    simplex[k-1] delta;
                    ordered[k-1] alpha;
                }
                model {
                    %s
                    {
                        vector[k-1] u;
                        for (i in 1:(k-1))
                            u[i] = 1;
                        delta ~ dirichlet(%f*u);
                    }
                    alpha ~ normal(0, %f);
                    for (i in 1:n)
                        y_post[i] ~ ordered_logistic((%s)*sum(delta[:y_pre[i]-1]), alpha);
                }
            '''%('\n'.join(code['data']), '\n'.join(code['parameters']), '\n'.join(code['model']), prior_delta, prior_alpha, ' + '.join(code['output']))
    
    mod_bs = '''
                data {
                    int<lower=1> n; //number of data points
                    int<lower=2> k; //number of outcomes
                    int<lower=1,upper=k> y_pre[n]; //pre-exposure
                    %s
                }
                parameters {
                    %s
                    ordered[k-1] alpha;
                }
                model {
                    %s
                    alpha ~ normal(0, %f);
                    for (i in 1:n)
                        y_pre[i] ~ ordered_logistic(%s, alpha);
                }
            '''%('\n'.join(code['data']), '\n'.join(code['parameters']), '\n'.join(code['model']), prior_alpha, ' + '.join(code['output']))
    
    data = {}
    if group is not None:
        df = df.loc[df['Treatment']==group]
        data['y_post'] = df['Vaccine Intent for %s (Post)'%kind].values
    data['n'] = df.shape[0]
    data['k'] = 4
    data['y_pre'] = df['Vaccine Intent for %s (Pre)'%kind].values        
    print('Dataframe of size:', df.shape)
    for i in range(len(cats)):
        name = dims[i].name
        data['k_%s'%name] = len(dd[cats[i]])
        data[name] = np.array(df[cats[i]].values, dtype=int)
        if data[name].min()==0: data[name] += 1
    if group is None: model = st.StanModel(model_code=mod_bs)
    else: model = st.StanModel(model_code=mod_cd)
    fit = model.sampling(data=data, iter=iters)
    return fit

def model_image_perceptions(df, group=1, prior_alpha=1., iters=NUM_SAMPLES):
    # Model: Model for self-reported image-metrics
    # Results: Figure 5
    import pystan as st
    import numpy as np
    model_code = '''
                    data {
                        int<lower=0> n; //number of data points
                        int<lower=2> k; //number of outcomes
                        int<lower=1,upper=k> y[n]; //outcome per sample
                    }
                    parameters {
                        ordered [k-1] alpha;
                    }
                    model {
                        alpha ~ normal(0, %f);
                        for (i in 1:n)
                            y[i] ~ ordered_logistic(0, alpha);
                    }
                '''%(prior_alpha)
    metrics = ['Vaccine Intent', 'Agreement', 'Trust', 'Fact-check', 'Share']
    fits = [dict() for i in range(5)]
    df = df.loc[df['Treatment']==group]
    for i in range(5):
        for m in metrics:
            data = {'n':df.shape[0], 'k':5, 'y':df['Image %i:%s'%(i+1, m)].values+3}
            model = st.StanModel(model_code=model_code)
            fits[i][m] = model.sampling(data=data, iter=iters)
    return fits

def model_image_impact(df, group=1, kind='self', prior_beta=1., prior_delta=1., prior_gamma=1., prior_alpha=1., iters=NUM_SAMPLES):
    # Model: Ref 6, Table 3
    # Results: Tables S6, S7
    import pystan as st
    import numpy as np
    model_code = '''
                    data {
                        int<lower=1> n; //number of data points
                        int<lower=1> p; //number of images
                        int<lower=1> m; //number of metrics
                        int<lower=2> k; //number of outcomes
                        int<lower=1,upper=k> y_pre[n]; //pre-exposure
                        int<lower=1,upper=k> y_post[n]; //post-exposure
                        matrix[p,m] x_img[n]; //image metrics
                    }
                    parameters {
                        vector[m] beta;
                        simplex[p] gamma;
                        simplex[k-1] delta;
                        ordered[k-1] alpha;
                    }
                    model {
                        beta ~ normal(0, %f);
                        {
                            vector[p] u_img;
                            for (i in 1:p)
                                u_img[i] = 1;
                            gamma ~ dirichlet(%f*u_img);
                        }
                        {
                            vector[k-1] u;
                            for (i in 1:(k-1))
                                u[i] = 1;
                            delta ~ dirichlet(%f*u);
                        }
                        alpha ~ normal(0, %f);
                        for (i in 1:n)
                            {
                                real b = to_row_vector(gamma)*x_img[i]*beta;
                                y_post[i] ~ ordered_logistic(b*sum(delta[:y_pre[i]-1]), alpha);
                            }
                    }
                '''%(prior_beta, prior_gamma, prior_delta, prior_alpha)
    
    metrics = ['Vaccine Intent', 'Agreement', 'Trust', 'Fact-check', 'Share']
    df = df.loc[df['Treatment']==group]
    x = np.dstack([df[['Image %i:%s'%(i+1, m) for i in range(5)]].values for m in metrics])
    data = {'n':df.shape[0], 'p':5, 'm':len(metrics), 'k':4, 'x_img':x,
            'y_pre':df['Vaccine Intent for %s (Pre)'%kind].values, 
            'y_post':df['Vaccine Intent for %s (Post)'%kind].values}
    model = st.StanModel(model_code=model_code)
    fit = model.sampling(data=data, iter=iters)
    return fit

def model_filterbubble(df, group=1, kind='self', prior_beta=1., prior_alpha=1., iters=NUM_SAMPLES):
    # Model: Model for evidence of filter-bubble effects of (mis)information exposure with regards to vaccine intent
    # Results: Table S8; Figure S3
    import pystan as st
    import numpy as np    
    model_code = '''
                    data {
                        int<lower=1> n; //number of data points
                        int<lower=1> m; //number of categories
                        int<lower=2> k; //number of outcomes
                        int<lower=1,upper=m> y_pre[n]; //pre-exposure
                        int<lower=1,upper=k> y_see[n]; //seen images
                    }
                    parameters {
                        real beta[m];
                        ordered[k-1] alpha;
                    }
                    model {
                        beta ~ normal(0, %f);
                        alpha ~ normal(0, %f);
                        for (i in 1:n)
                            y_see[i] ~ ordered_logistic(beta[y_pre[i]], alpha);
                    }
                '''%(prior_beta, prior_alpha)
    
    df = df.loc[(df['Treatment']==group) & (df['Seen such online content']!=3)] #ignoring do-not-know's
    data = {'n':df.shape[0], 'm':4, 'k':2,
            'y_pre':df['Vaccine Intent for %s (Pre)'%kind].values,
            'y_see':[i%2+1 for i in df['Seen such online content'].values]} #"yes":2, "no":1 for ordinal logit
    model = st.StanModel(model_code=model_code)
    fit = model.sampling(data=data, iter=iters)
    return fit