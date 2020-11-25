NUM_SAMPLES = 2000 #number of samples per chain; 4 chains total
ADAPT_DELTA = 0.9 #target Metropolis acceptance rate; larger for finer sampling
M_TREEDEPTH = 10

def model_impact(df, group=1., kind='self', prior_mu=1., prior_alpha=1., iters=NUM_SAMPLES, adapt_delta=ADAPT_DELTA, max_treedepth=M_TREEDEPTH):
    # Model: Ref 1, Table 3
    # Results: Table 2
    import pystan as st
    model_code = '''
                    data {
                        int<lower=1> n; //number of data points
                        int<lower=1> m; //number of conditions
                        int<lower=2> k; //number of outcomes
                        int<lower=1,upper=k> y[n,m]; //outcome per sample per condition
                    }
                    parameters {
                        vector[k-1] mu;
                        ordered[k-1] alpha[m];
                    }
                    model {
                        mu ~ normal(0, %f);
                        for (i in 1:m)
                            alpha[i] ~ normal(mu, %f);
                        for (i in 1:n)
                            for (j in 1:m)
                                y[i,j] ~ ordered_logistic(0, alpha[j]);
                    }
                '''%(prior_mu, prior_alpha)
    df = df.loc[df['Treatment']==group]
    data = {'n':df.shape[0], 'm':2, 'k':4, 'y':df[['Vaccine Intent for %s (Pre)'%kind, 'Vaccine Intent for %s (Post)'%kind]].values}
    model = st.StanModel(model_code=model_code)
    fit = model.sampling(data=data, iter=iters, control=dict(adapt_delta=adapt_delta, max_treedepth=max_treedepth))
    return fit

def model_impact_causal(df, kind='self', prior_beta=1., prior_delta=1., prior_alpha=1., prior_mu_beta=1., prior_mu_alpha=1., iters=NUM_SAMPLES, adapt_delta=ADAPT_DELTA, max_treedepth=M_TREEDEPTH):
    # Model: Appendix C
    # Results: Tables S1, S2; Figures S1, S2
    import pystan as st
    model_code = '''
                data {
                    int<lower=1> n; //number of data points
                    int<lower=1> m; //number of conditions
                    int<lower=2> k; //number of outcomes
                    int<lower=1,upper=m> x_cond[n]; //treatment group
                    int<lower=1,upper=k> y_pre[n]; //pre-exposure outcome
                    int<lower=1,upper=k> y_post[n]; //post-exposure outcome
                }
                parameters {
                    real mu_beta;
                    real beta[m];
                    vector<lower=0>[k-1] mu_delta;
                    simplex[k-1] delta[m];
                    vector[k-1] mu_alpha;
                    ordered[k-1] alpha[m];
                }
                model {
                    mu_beta ~ normal(0, %f);
                    mu_alpha ~ normal(0, %f);
                    mu_delta ~ exponential(%f);
                    for (i in 1:m){
                        beta[i] ~ normal(mu_beta, %f);
                        alpha[i] ~ normal(mu_alpha, %f);
                        delta[i] ~ dirichlet(mu_delta);
                    }
                    for (i in 1:n)
                        y_post[i] ~ ordered_logistic(beta[x_cond[i]]*sum(delta[x_cond[i]][:y_pre[i]-1]), alpha[x_cond[i]]);
                }
            '''%(prior_mu_beta, prior_mu_alpha, prior_delta, prior_beta, prior_alpha)
    
    data = {'n':df.shape[0], 'm':2, 'k':4, 'x_cond':df['Treatment'].values+1, 
            'y_pre':df['Vaccine Intent for %s (Pre)'%kind].values, 
            'y_post':df['Vaccine Intent for %s (Post)'%kind].values}
    model = st.StanModel(model_code=model_code)
    fit = model.sampling(data=data, iter=iters, control=dict(adapt_delta=adapt_delta, max_treedepth=max_treedepth))
    return fit

def model_socdem(df, dd, atts=[], causal=True, group=None, kind='self', decrease=0, prior_beta=1., prior_delta=1., prior_alpha=1., prior_mu_beta=1., prior_mu_alpha=1., iters=NUM_SAMPLES, adapt_delta=ADAPT_DELTA, max_treedepth=M_TREEDEPTH):
    # Model: Ref 2, 3, 4, 5 in Table 3
    # Results: Tables S3, S4, S5; Figures 3, 4
    import pystan as st
    import numpy as np
    from .bayesoc import Dim #we define some helper classes to extract posterior samples easily
    cats = ['Age', 'Gender', 'Education', 'Employment', 'Religion', 'Political', 'Ethnicity', 'Income']
    if isinstance(atts, str): atts = [atts]
    for att in atts: cats += [x for x in list(df) if x[:len(att)]==att]
    outs = ['Vaccine Intent for self (Pre)', 'Vaccine Intent for self (Post)', 'Treatment']
    df = df[cats+outs].dropna()
    dims = [Dim(pi=len(dd[cat]), out=int(causal)+1, prior_beta=prior_beta, prior_mu_beta=prior_mu_beta, value=dd[cat].keys(), name=cat) for cat in cats]
    stan = [d.get_stan(outcome_size='m', outcome_index='x_cond[i]', hierarchical=True) for d in dims]
    code = {'data':[], 'parameters':[], 'model':[], 'output':[]}
    for key in code:
        for d in stan: code[key].append(d[key])
    model_code = {
                    'pre': '''
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
                        '''%('\n'.join(code['data']), '\n'.join(code['parameters']), '\n'.join(code['model']), prior_alpha, ' + '.join(code['output'])),
                    'post': '''
                            data {
                                int<lower=1> n; //number of data points
                                int<lower=2> k; //number of outcomes
                                int<lower=1,upper=k> y_pre[n]; //pre-exposure
                                int<lower=1,upper=k> y_post[n]; //post-exposure
                                %s
                            }
                            parameters {
                                %s
                                simplex[k-1-%i] delta;
                                ordered[k-1] alpha;
                            }
                            model {
                                %s
                                {
                                    vector[k-1-%i] u;
                                    for (i in 1:(k-1-%i))
                                        u[i] = 1;
                                    delta ~ dirichlet(%f*u);
                                }
                                alpha ~ normal(0, %f);
                                for (i in 1:n)
                                    y_post[i] ~ ordered_logistic((%s)*sum(delta[:y_pre[i]-1-%i]), alpha);
                            }
                        '''%('\n'.join(code['data']), '\n'.join(code['parameters']), decrease, '\n'.join(code['model']), decrease, decrease, prior_delta, prior_alpha, ' + '.join(code['output']), decrease),
                    'causal' : '''
                            data {
                                int<lower=1> n; //number of data points
                                int<lower=1> m; //number of conditions
                                int<lower=2> k; //number of outcomes
                                int<lower=1,upper=m> x_cond[n]; //treatment group
                                int<lower=1,upper=k> y_pre[n]; //pre-exposure
                                int<lower=1,upper=k> y_post[n]; //post-exposure
                                %s
                            }
                            parameters {
                                %s
                                vector<lower=0>[k-1-%i] mu_delta;
                                simplex[k-1-%i] delta[m];
                                vector[k-1] mu_alpha;
                                ordered[k-1] alpha[m];
                            }
                            model {
                                %s
                                mu_delta ~ exponential(%f);
                                mu_alpha ~ normal(0, %f); 
                                for (i in 1:m){
                                    delta[i] ~ dirichlet(mu_delta);
                                    alpha[i] ~ normal(mu_alpha, %f);
                                }
                                for (i in 1:n)
                                    y_post[i] ~ ordered_logistic((%s)*sum(delta[x_cond[i]][:y_pre[i]-1-%i]), alpha[x_cond[i]]);
                            }
                        '''%('\n'.join(code['data']), '\n'.join(code['parameters']), decrease, decrease, '\n'.join(code['model']), prior_delta, prior_mu_alpha, prior_alpha, ' + '.join(code['output']), decrease)
                }
    data = {}
    if causal:
        data['m'] = 2
        data['x_cond'] = df['Treatment'].values+1
        data['y_post'] = df['Vaccine Intent for %s (Post)'%kind].values
    elif group is not None:
        df = df.loc[df['Treatment']==group]
        data['y_post'] = df['Vaccine Intent for %s (Post)'%kind].values
    data['n'] = df.shape[0]
    data['k'] = 4 - int(decrease*int(group is None and not causal))
    data['y_pre'] = df['Vaccine Intent for %s (Pre)'%kind].values - int(decrease*int(group is None and not causal))
    print('Dataframe of size:', df.shape)
    for i in range(len(cats)):
        name = dims[i].name
        data['k_%s'%name] = len(dd[cats[i]])
        data[name] = np.array(df[cats[i]].values, dtype=int)
        if data[name].min()==0: data[name] += 1
    if causal: model_name = 'causal'
    elif group is None: model_name = 'pre'
    else: model_name = 'post'
    model = st.StanModel(model_code=model_code[model_name])
    fit = model.sampling(data=data, iter=iters, control=dict(adapt_delta=adapt_delta, max_treedepth=max_treedepth))
    return fit

def model_image_perceptions(df, group=1, prior_alpha=1., iters=NUM_SAMPLES, adapt_delta=ADAPT_DELTA, max_treedepth=M_TREEDEPTH):
    # Model: Model for self-reported image-metrics
    # Results: Figure 5
    import pystan as st
    import numpy as np
    model_code = '''
                    data {
                        int<lower=1> n; //number of data points
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
            fits[i][m] = model.sampling(data=data, iter=iters, control=dict(adapt_delta=adapt_delta, max_treedepth=max_treedepth))
    return fits

def model_image_impact(df, group=1, kind='self', prior_beta=1., prior_delta=1., prior_gamma=1., prior_alpha=1., iters=NUM_SAMPLES, adapt_delta=ADAPT_DELTA, max_treedepth=M_TREEDEPTH):
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
    fit = model.sampling(data=data, iter=iters, control=dict(adapt_delta=adapt_delta, max_treedepth=max_treedepth))
    return fit

def model_filterbubble(df, group=1, kind='self', prior_beta=1., prior_alpha=1., iters=NUM_SAMPLES, adapt_delta=ADAPT_DELTA, max_treedepth=M_TREEDEPTH):
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
    fit = model.sampling(data=data, iter=iters, control=dict(adapt_delta=adapt_delta, max_treedepth=max_treedepth))
    return fit