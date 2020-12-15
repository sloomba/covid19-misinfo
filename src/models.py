NUM_SAMPLES = 2000 #number of samples per chain; 4 chains total
ADAPT_DELTA = 0.9 #target Metropolis acceptance rate; larger for finer sampling
M_TREEDEPTH = 10

def model_impact_causal(df, kind='self', prior_beta=1., prior_delta=1., prior_alpha=1., prior_mu_beta=1., prior_mu_alpha=1., iters=NUM_SAMPLES, adapt_delta=ADAPT_DELTA, max_treedepth=M_TREEDEPTH):
    # Model: Eqs. 1 & 2, with `f=0`
    # Results: Tables 1, S2; Figure 4
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
                    ordered[k-1] alpha_pre;
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
                    alpha_pre ~ normal(0, %f);
                    for (i in 1:n){
                        y_pre[i] ~ ordered_logistic(0, alpha_pre);
                        y_post[i] ~ ordered_logistic(beta[x_cond[i]]*sum(delta[x_cond[i]][:y_pre[i]-1]), alpha[x_cond[i]]);
                    }
                }
            '''%(prior_mu_beta, prior_mu_alpha, prior_delta, prior_beta, prior_alpha, prior_alpha)
    
    data = {'n':df.shape[0], 'm':2, 'k':4, 'x_cond':df['Treatment'].values+1, 
            'y_pre':df['Vaccine Intent for %s (Pre)'%kind].values, 
            'y_post':df['Vaccine Intent for %s (Post)'%kind].values}
    model = st.StanModel(model_code=model_code)
    fit = model.sampling(data=data, iter=iters, control=dict(adapt_delta=adapt_delta, max_treedepth=max_treedepth))
    return fit

def model_socdem(df, dd, atts=[], model_name='causal', group=None, kind='self', decrease=0, prior_beta=1., prior_delta=1., prior_alpha=1., prior_mu_beta=1., prior_mu_alpha=1., iters=NUM_SAMPLES, adapt_delta=ADAPT_DELTA, max_treedepth=M_TREEDEPTH):
    # Model: Eqs. 1 & 2, with `f` modeling a linear combination of socio-demographics and other covariates
    # Results: Tables S3, S4, S5, S6; Figures 5, S1, S2, S3
    import pystan as st
    import numpy as np
    from .bayesoc import Dim #we define some helper classes to extract posterior samples easily
    cats = ['Age', 'Gender', 'Education', 'Employment', 'Religion', 'Political', 'Ethnicity', 'Income']
    if isinstance(atts, str): atts = [atts]
    for att in atts: cats += [x for x in list(df) if x[:len(att)]==att]
    outs = ['Vaccine Intent for %s (Pre)'%kind, 'Vaccine Intent for %s (Post)'%kind, 'Treatment']
    df = df[cats+outs].dropna()
    if group is not None: model_name='post'
    causal = int(model_name in ['causal', 'causaldiff'])
    dims = [Dim(pi=len(dd[cat]), out=causal+1, prior_beta=prior_beta, prior_mu_beta=prior_mu_beta, value=dd[cat].keys(), name=cat) for cat in cats]
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
                                real beta;
                                simplex[k-1-%i] delta;
                                ordered[k-1] alpha;
                            }
                            model {
                                %s
                                beta ~ normal(0, %f);
                                {
                                    vector[k-1-%i] u;
                                    for (i in 1:(k-1-%i))
                                        u[i] = 1;
                                    delta ~ dirichlet(%f*u);
                                }
                                alpha ~ normal(0, %f);
                                for (i in 1:n)
                                    y_post[i] ~ ordered_logistic(beta*sum(delta[:y_pre[i]-1-%i]) + %s, alpha);
                            }
                        '''%('\n'.join(code['data']), '\n'.join(code['parameters']), decrease, '\n'.join(code['model']), prior_beta, decrease, decrease, prior_delta, prior_alpha, decrease, ' + '.join(code['output'])),
                    'causal' : '''
                            data {
                                int<lower=1> n; //number of data points
                                int<lower=1> m; //number of conditions
                                int<lower=2> k; //number of outcomes
                                int<lower=1,upper=m> x_cond[n]; //treatment group
                                int<lower=1,upper=k> y_pre[n]; //pre-exposure outcome
                                int<lower=1,upper=k> y_post[n]; //post-exposure outcome
                                %s
                            }
                            parameters {
                                %s
                                real mu_beta;
                                real beta[m];
                                vector<lower=0>[k-1-%i] mu_delta;
                                simplex[k-1-%i] delta[m];
                                vector[k-1] mu_alpha;
                                ordered[k-1] alpha[m];
                            }
                            model {
                                %s
                                mu_beta ~ normal(0, %f);
                                mu_alpha ~ normal(0, %f);
                                mu_delta ~ exponential(%f);
                                for (i in 1:m){
                                    beta[i] ~ normal(mu_beta, %f);
                                    alpha[i] ~ normal(mu_alpha, %f);
                                    delta[i] ~ dirichlet(mu_delta);
                                }
                                for (i in 1:n)
                                    y_post[i] ~ ordered_logistic(beta[x_cond[i]]*sum(delta[x_cond[i]][:y_pre[i]-1-%i]) + %s, alpha[x_cond[i]]);
                            }
                        '''%('\n'.join(code['data']), '\n'.join(code['parameters']), decrease, decrease, '\n'.join(code['model']), prior_mu_beta, prior_mu_alpha, prior_delta, prior_beta, prior_alpha, decrease, ' + '.join(code['output'])),
                    'causaldiff' : '''
                            data {
                                int<lower=1> n; //number of data points
                                int<lower=1> m; //number of conditions
                                int<lower=2> k; //number of outcomes
                                int<lower=1,upper=m> x_cond[n]; //treatment group
                                int<lower=1,upper=k> y_diff[n]; //post-pre difference in outcomes
                                %s
                            }
                            parameters {
                                %s
                                vector[k-1] mu_alpha;
                                ordered[k-1] alpha[m];
                            }
                            model {
                                %s
                                mu_alpha ~ normal(0, %f); 
                                for (i in 1:m)
                                    alpha[i] ~ normal(mu_alpha, %f);
                                for (i in 1:n)
                                    y_diff[i] ~ ordered_logistic(%s, alpha[x_cond[i]]);
                            }
                        '''%('\n'.join(code['data']), '\n'.join(code['parameters']), '\n'.join(code['model']), prior_mu_alpha, prior_alpha, ' + '.join(code['output']))
                }
    data = {}
    if causal:
        data['m'] = 2
        data['x_cond'] = df['Treatment'].values+1
        if model_name=='causal':
            data['k'] = 4
            data['y_pre'] = df['Vaccine Intent for %s (Pre)'%kind].values
            data['y_post'] = df['Vaccine Intent for %s (Post)'%kind].values
        else:
            data['k'] = 3
            tmp = df['Vaccine Intent for %s (Post)'%kind].values - df['Vaccine Intent for %s (Pre)'%kind].values
            tmp[tmp>0] = 1
            tmp[tmp<0] = -1
            data['y_diff'] = tmp+2
    elif model_name=='post':
        data['k'] = 4
        df = df.loc[df['Treatment']==group]
        data['y_pre'] = df['Vaccine Intent for %s (Pre)'%kind].values
        data['y_post'] = df['Vaccine Intent for %s (Post)'%kind].values
    else:
        data['k'] = 4 - decrease
        data['y_pre'] = df['Vaccine Intent for %s (Pre)'%kind].values - decrease
    data['n'] = df.shape[0]
    print('Dataframe of size:', df.shape)
    for i in range(len(cats)):
        name = dims[i].name
        data['k_%s'%name] = len(dd[cats[i]])
        data[name] = np.array(df[cats[i]].values, dtype=int)
        if data[name].min()==0: data[name] += 1
    model = st.StanModel(model_code=model_code[model_name])
    fit = model.sampling(data=data, iter=iters, control=dict(adapt_delta=adapt_delta, max_treedepth=max_treedepth))
    return fit

def model_image_impact(df, group=1, kind='self', prior_beta=1., prior_delta=1., prior_gamma=1., prior_alpha=1., iters=NUM_SAMPLES, adapt_delta=ADAPT_DELTA, max_treedepth=M_TREEDEPTH):
    # Model: Eq. 7
    # Results: Table S7
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
                        real beta;
                        vector[m] beta_img;
                        simplex[p] gamma;
                        simplex[k-1] delta;
                        ordered[k-1] alpha;
                    }
                    model {
                        beta ~ normal(0, %f);
                        beta_img ~ normal(0, %f);
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
                            y_post[i] ~ ordered_logistic(beta*sum(delta[:y_pre[i]-1]) + to_row_vector(gamma)*x_img[i]*beta_img, alpha);
                    }
                '''%(prior_beta, prior_beta, prior_gamma, prior_delta, prior_alpha)
    
    metrics = ['Vaccine Intent', 'Agreement', 'Trust', 'Fact-check', 'Share']
    df = df.loc[df['Treatment']==group]
    x = np.dstack([df[['Image %i:%s'%(i+1, m) for i in range(5)]].values for m in metrics])
    data = {'n':df.shape[0], 'p':5, 'm':len(metrics), 'k':4, 'x_img':x,
            'y_pre':df['Vaccine Intent for %s (Pre)'%kind].values, 
            'y_post':df['Vaccine Intent for %s (Post)'%kind].values}
    model = st.StanModel(model_code=model_code)
    fit = model.sampling(data=data, iter=iters, control=dict(adapt_delta=adapt_delta, max_treedepth=max_treedepth))
    return fit

def model_similar_content(df, model_name='seen', kind='self', prior_beta=1., prior_alpha=1., prior_delta=1., prior_mu_beta=1., prior_mu_alpha=1., iters=NUM_SAMPLES, adapt_delta=ADAPT_DELTA, max_treedepth=M_TREEDEPTH):
    import pystan as st
    import numpy as np
    model_code = {
                    'pre':'''
                            data {
                                int<lower=1> n; //number of data points
                                int<lower=1> m; //number of conditions
                                int<lower=2> k; //number of outcomes
                                int<lower=1,upper=m> x_cond[n]; //treatment group
                                int<lower=0,upper=1> x_seen[n]; //seen images
                                int<lower=1,upper=k> y_pre[n]; //pre-exposure outcome
                            }
                            parameters {
                                real mu_beta;
                                real beta[m];
                                vector[k-1] mu_alpha;
                                ordered[k-1] alpha[m];
                                real<lower=0,upper=1> theta[m];
                            }
                            model {
                                mu_beta ~ normal(0, %f);
                                mu_alpha ~ normal(0, %f);
                                for (i in 1:m){
                                    beta[i] ~ normal(mu_beta, %f);
                                    alpha[i] ~ normal(mu_alpha, %f);
                                }
                                theta ~ beta(1, 1);
                                for (i in 1:n){
                                    x_seen[i] ~ bernoulli(theta[x_cond[i]]);
                                    y_pre[i] ~ ordered_logistic(beta[x_cond[i]]*x_seen[i], alpha[x_cond[i]]);
                                }
                            }
                        '''%(prior_mu_beta, prior_mu_alpha, prior_beta, prior_alpha),
                    'causal':'''
                            data {
                                int<lower=1> n; //number of data points
                                int<lower=1> m; //number of conditions
                                int<lower=2> k; //number of outcomes
                                int<lower=1,upper=m> x_cond[n]; //treatment group
                                int<lower=0,upper=1> x_seen[n]; //seen images
                                int<lower=1,upper=k> y_pre[n]; //pre-exposure outcome
                                int<lower=1,upper=k> y_post[n]; //post-exposure outcome
                            }
                            parameters {
                                real mu_beta;
                                real beta[m];
                                real mu_beta_pre;
                                real beta_pre[m];
                                real mu_beta_post;
                                real beta_post[m];
                                vector<lower=0>[k-1] mu_delta;
                                simplex[k-1] delta[m];
                                vector[k-1] mu_alpha_pre;
                                ordered[k-1] alpha_pre[m];
                                vector[k-1] mu_alpha_post;
                                ordered[k-1] alpha_post[m];
                                real<lower=0,upper=1> theta[m];
                            }
                            model {
                                mu_beta ~ normal(0, %f);
                                mu_beta_pre ~ normal(0, %f);
                                mu_beta_post ~ normal(0, %f);
                                mu_alpha_pre ~ normal(0, %f);
                                mu_alpha_post ~ normal(0, %f);
                                mu_delta ~ exponential(%f);
                                for (i in 1:m){
                                    beta[i] ~ normal(mu_beta, %f);
                                    beta_pre[i] ~ normal(mu_beta_pre, %f);
                                    beta_post[i] ~ normal(mu_beta_post, %f);
                                    alpha_pre[i] ~ normal(mu_alpha_pre, %f);
                                    alpha_post[i] ~ normal(mu_alpha_post, %f);
                                    delta[i] ~ dirichlet(mu_delta);
                                }
                                theta ~ beta(1, 1);
                                for (i in 1:n){
                                    x_seen[i] ~ bernoulli(theta[x_cond[i]]);
                                    y_pre[i] ~ ordered_logistic(beta_pre[x_cond[i]]*x_seen[i], alpha_pre[x_cond[i]]);
                                    y_post[i] ~ ordered_logistic(beta[x_cond[i]]*sum(delta[x_cond[i]][:y_pre[i]-1])+beta_post[x_cond[i]]*x_seen[i], alpha_post[x_cond[i]]);
                                }
                            }
                        '''%(prior_mu_beta, prior_mu_beta, prior_mu_beta, prior_mu_alpha, prior_mu_alpha, prior_delta, prior_beta, prior_beta, prior_beta, prior_alpha, prior_alpha),
                    'seen_ordinal':'''
                            data {
                                int<lower=1> n; //number of data points
                                int<lower=1> m; //number of conditions
                                int<lower=2> k; //number of outcomes
                                int<lower=1,upper=m> x_cond[n]; //treatment group
                                int<lower=0,upper=1> x_seen[n]; //seen images
                                int<lower=1,upper=k> y_pre[n]; //pre-exposure outcome
                            }
                            parameters {
                                real mu_beta;
                                real beta[m];
                                vector<lower=0>[k-1] mu_delta;
                                simplex[k-1] delta[m];
                                real mu_alpha;
                                real alpha[m];
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
                                    x_seen[i] ~ bernoulli_logit(beta[x_cond[i]]*sum(delta[x_cond[i]][:y_pre[i]-1]) + alpha[x_cond[i]]);
                            }
                        '''%(prior_mu_beta, prior_mu_alpha, prior_delta, prior_beta, prior_alpha),
                    'seen':'''
                            data {
                                int<lower=1> n; //number of data points
                                int<lower=1> m; //number of conditions
                                int<lower=2> k; //number of outcomes
                                int<lower=1,upper=m> x_cond[n]; //treatment group
                                int<lower=0,upper=1> x_seen[n]; //seen images
                                int<lower=1,upper=k> y_pre[n]; //pre-exposure outcome
                            }
                            parameters {
                                vector[k] mu_beta;
                                vector[k] beta[m];
                            }
                            model {
                                mu_beta ~ normal(0, %f);
                                for (i in 1:m)
                                    beta[i] ~ normal(mu_beta, %f);
                                for (i in 1:n)
                                    x_seen[i] ~ bernoulli_logit(beta[x_cond[i]][y_pre[i]]);
                            }
                        '''%(prior_mu_beta, prior_beta)
                }

    df = df.loc[df['Seen such online content']!=3] #ignoring do-not-know's
    data = {'n':df.shape[0], 'm':2, 'k':4, 'x_cond':df['Treatment'].values+1,
            'y_pre':df['Vaccine Intent for %s (Pre)'%kind].values,
            'x_seen':[i%2 for i in df['Seen such online content'].values]} #"yes":1, "no":0
    if model_name=='causal': data['y_post'] = df['Vaccine Intent for %s (Post)'%kind].values
    model = st.StanModel(model_code=model_code[model_name])
    fit = model.sampling(data=data, iter=iters, control=dict(adapt_delta=adapt_delta, max_treedepth=max_treedepth))
    return fit