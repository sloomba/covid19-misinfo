def import_datadict(datadir='./dat', filename='orb_datadict.txt'):
    '''Imports the data dictionary for raw survey data.
    See questions and field names in /doc/orb_questionnaire.pdf

    Args:
        datadir (:obj:`str`): name of directory where dictionary is located
        filename (:obj:`str`): name of dictionary file (.txt)

    Returns:
        A :obj:`dict` containing keys:
        * 'dict' which maps to a :obj:`dict` of field names (:obj:`str') mapped to a :obj:`dict` of values (:obj:`str`) mapped to recoded-vaues (:obj:`int`)
        * 'desc' which maps to a :obj:`dict` of field names (:obj:`str') mapped to their descriptions (:obj:`str`)
    '''
    import os
    filepath = os.path.join(datadir, filename)
    with open(filepath, encoding='utf-8') as fd: data = fd.readlines()
    data = [d for d in data if d.strip()]
    for i in range(len(data)):
        if data[i][0]!='\t': data[i] = [x.strip() for x in data[i].split(':')]
        else: data[i] = [data[i].strip()]
    dmap = {}
    text = {}
    curr = ''
    multi = ''
    for d in data:
        if len(d)==1:
            tmp = d[0].split('\t')
            if len(tmp)==2:
                if tmp[0][0]=='[':
                    curr = tmp[0][1:-1]
                    desc = tmp[1].strip()
                    text[curr] = desc
                    dmap[curr] = dmap[multi].copy()
                else:
                    dmap[curr][tmp[1]] = int(tmp[0])
        elif len(d)>1:
            if d[0][0]=='[':
                curr = d[0][1:-1]
                desc = d[1].strip()
                text[curr] = desc
                dmap[curr] = {}
            elif d[0]!='Values':
                curr = d[0]
                desc = d[1].strip()
                text[curr] = desc
                dmap[curr] = {}
                multi = curr
    #rectify some encoding issues
    errata = {'DREL':{'Other Christian:':'Other Christian:\xa0'},
              'DPOLUK':{'Other:':'Other:\xa0'}
                }
    #recoding extra variables of age categories and treatment (ANTI) or control (PRO) group
    extras = {'imageseen':{'ANTI US':1, 'PRO US':0, 'ANTI UK':1, 'PRO UK':0}, 
              'agerecode':{ '18-24':1, '25-34':2, '35-44':3, '45-54':4, '55-64':5, '65+':6}}
    for key in errata:
        if key in dmap:
            for label in errata[key]:
                if label in dmap[key]:
                    dmap[key][errata[key][label]] = dmap[key][label]
                    del dmap[key][label]
    for key in extras: dmap[key] = extras[key]
    return {'dict':dmap,'desc':text}

def import_data(datadir='./dat', filename='orb_200918.sav'):
    '''Reads a survey SPSS file and returns a :obj:`pd.DataFrame`.
    See questions and field names in /doc/orb_questionnaire.pdf

    Args:
        datadir (:obj:`str`): name of directory where SPSS file is located
        filename (:obj:`str`): name of the SPSS file (.sav)

    Returns:
        A :obj:`pd.DataFrame` containing field names as columns and record as rows, with values recoded 
    '''
    import os
    import pandas as pd
    filepath = os.path.join(datadir, filename)
    if filepath[-4:] == '.sav': df = pd.read_spss(filepath)
    else: df = pd.read_csv(filepath)
    for att in list(df):
        try: df[att].str.strip('\xa0') #funny encoding issue
        except: pass
    return df

def transform_data(df, dd, country='UK', group=None, save=''):
    '''Cleans, recodes and transforms raw survey data.
    See questions and field names in /doc/orb_questionnaire.pdf

    Args:
        df (:obj:`pd.DataFrame`): contains raw survey data (see return value of ``import_data()``)
        dd (:obj:`dict`): contains data dictionary for the raw survey data (see return value of ``import_datadict()``)
        country (:obj:`str`=`{'UK', 'US'}`: name of country of interest; default=`UK`
        group (:obj:`int`=`{0, 1}`): name of the experiment group, where `0` is for control and `1` is for treatment; default=`None` (imports all samples)
        save (:obj:`str`): filepath to save the processed data (as a .csv) and data dictionary (as a .pkl); default='' (does not save anything)

    Returns:
        A size-2 :obj:`tuple` containing
        * A :obj:`pd.DataFrame` containing field names as columns and record as rows of the transformed data
        * A :obj:`dict` of field names (:obj:`str`) mapped to a :obj:`dict` of recoded-values (:obj:`int`) mapped to value-names (:obj:`str`)
    '''
    #define all socio-demographic variables of interest
    demo = {'UK': {'agerecode':'Age', 'DGEN':'Gender', 'DEDUUK':'Education_UK', 'DEMP':'Employment', 'DREL':'Religion', 
                  'DPOLUK':'Political_UK', 'DETHUK':'Ethnicity_UK', 'DINCUK':'Income_UK', 'DGEOUK':'Region'},
            'USA': {'agerecode':'Age', 'DGEN':'Gender', 'DEDUUS':'Education_US', 'DEMP':'Employment', 'DREL':'Religion',
                    'DPOLUS':'Political_US', 'DETHUS':'Ethnicity_US', 'DINCUS':'Income_US', 'DGEOUS':'Region'}}
    
    #define recoding of socio-demographics
    var_encoding = {'Gender':{(1,):'Male', (2,):'Female', (3, 4):'Other'},
                    'Education_US':{(1, 2):'Level-0', (3,):'Level-1', (4,):'Level-2', (5,):'Level-3', (6,):'Level-4', (7, 8):'Other'}, 
                    'Education_UK':{(1,):'Level-0', (2, 3):'Level-1', (5,):'Level-2', (6,):'Level-3', (7,):'Level-4', (4, 8, 9):'Other'}, 
                    'Employment':{(1, 2):'Employed', (3,):'Unemployed', (4,):'Student', (6,):'Retired', (5, 7, 8):'Other'},
                    'Religion':{(1, 2, 3):'Christian', (4,):'Jewish', (6,):'Muslim', (9,):'Atheist', (5, 7, 8, 10):'Other'},
                    'Political_US':{(1,):'Republican', (2,):'Democrat', (3, 4, 5):'Other'},
                    'Political_US_ind':{(1,):'Republican', (2,):'Democrat', (3, 4):'Other'},
                    'Political_UK':{(1,):'Conservative', (2,):'Labour', (3,):'Liberal-Democrat', (4,):'SNP', (5,6,7):'Other'},
                    'Ethnicity_US':{(1,):'White', (2,):'Hispanic', (3,):'Black', (5,):'Asian', (4, 6, 7, 8):'Other'},
                    'Ethnicity_UK':{(1, 2, 3):'White', (4, 11):'Black', (5, 6, 7, 8, 9, 10):'Asian', (12, 13):'Other'},
                    'Income_US':{(1,):'Level-0', (2, 3):'Level-1', (4, 5): 'Level-2', (6, 7, 8, 9):'Level-3', (10,):'Level-4', (11,):'Other'},
                    'Income_UK':{(1,):'Level-0', (2,):'Level-1', (3,):'Level-2', (4, 5,):'Level-3', (6, 7, 8, 9, 10):'Level-4', (11,):'Other'}
                   }
    
    #rename other survey variables of interest to make them human-comprehendable
    metrics_any = {'QINFr1': 'Nobody', 'QINFr2': 'Myself', 'QINFr3': 'Family inside HH', 'QINFr4': 'Family outside HH', 
                   'QINFr5': 'Close friend', 'QINFr6': 'Colleague'}    
    metrics_knl = {'QKNLr1': 'Washing hands', 'QKNLr2': 'Staying indoors for Self', 'QKNLr3': 'Staying indoors for Others', 
                   'QKNLr4': 'Spread before symptoms', 'QKNLr5': 'R-Number', 'QKNLr6': 'Treatments already exist', 'QKNLr7': 'Wearing masks'}
    metrics_cov = {'QCOVVCIr3': 'COVID-19 Vax Importance', 'QCOVVCIr1': 'COVID-19 Vax Safety', 'QCOVVCIr2': 'COVID-19 Vax Efficacy', 
                   'QCOVVCIr4': 'COVID-19 Vax Compatibility', 'QCOVVCIr5': 'Contract via COVID-19 Vax', 'QCOVVCIr6': 'COVID-19 Vax benefits outweigh risks'}
    metrics_vci = {'QVCIr1': 'Vax Importance', 'QVCIr2': 'Vax Safety', 'QVCIr3': 'Vax Efficacy', 'QVCIr4': 'Vax Compatibility'}
    metrics_aff = {'QCOVAFFr1': 'Mental health', 'QCOVAFFr2': 'Financial stability', 'QCOVAFFr3': 'Daily disruption', 'QCOVAFFr4': 'Social disruption'}
    trust = {'UK': {'QSRCUKr1': 'Television', 'QSRCUKr2': 'Radio', 'QSRCUKr3': 'Newspapers', 'QSRCUKr4': 'Govt. Briefings', 
                    'QSRCUKr5': 'National Health Authorities', 'QSRCUKr6': 'International Health Authorities', 'QSRCUKr7': 'Healthcare Workers', 
                    'QSRCUKr8': 'Scientists', 'QSRCUKr9': 'Govt. Websites', 'QSRCUKr10': 'Social Media', 'QSRCUKr11': 'Celebrities', 'QSRCUKr12': 'Search Engines', 
                    'QSRCUKr13': 'Family and friends', 'QSRCUKr14': 'Work Guidelines', 'QSRCUKr15': 'Other', 'QSRCUKr16': 'None of these'},
             'USA': {'QSRCUSr1': 'Television', 'QSRCUSr2': 'Radio', 'QSRCUSr3': 'Newspapers', 'QSRCUSr4': 'White House Briefings', 'QSRCUSr5':'State Govt. Briefings', 
                     'QSRCUSr6': 'National Health Authorities', 'QSRCUSr7': 'International Health Authorities', 'QSRCUSr8':'Healthcare Workers', 
                     'QSRCUSr9': 'Scientists', 'QSRCUSr10': 'Govt. Websites', 'QSRCUSr11': 'Social Media', 'QSRCUSr12': 'Celebrities', 'QSRCUSr13': 'Search Engines', 
                     'QSRCUSr14': 'Family and friends', 'QSRCUSr15': 'Work Guidelines', 'QSRCUSr16': 'Other', 'QSRCUSr17': 'None of these'}}
    reasons = {'QCOVSELFWHYr1': 'Unsure if safe', 'QCOVSELFWHYr2': 'Unsure if effective', 'QCOVSELFWHYr3': 'Not at risk', 'QCOVSELFWHYr4': 'Wait until others',
               'QCOVSELFWHYr5': "Won't be ill", 'QCOVSELFWHYr6': 'Other effective treatments', 'QCOVSELFWHYr7': 'Already acquired immunity',
               'QCOVSELFWHYr8': 'Approval may be rushed', 'QCOVSELFWHYr9': 'Other', 'QCOVSELFWHYr10': 'Do not know'}
    metrics_img = {'QPOSTVACX_Lr': 'Vaccine Intent', 'QPOSTBELIEFX_Lr': 'Agreement', 'QPOSTTRUSTX_Lr': 'Trust', 
                   'QPOSTCHECKX_Lr': 'Fact-check', 'QPOSTSHARE_Lr': 'Share'}
    social_atts = {'QSOCTYPr': 'used', 'QSOCINFr': 'to receive info', 'QCIRSHRr': 'to share info'}
    other_atts = {'QSHD':'Shielding',
                  'QSOCUSE':'Social media usage', 
                  'QCOVWHEN':'Expected vax availability',
                  'QPOSTSIM':'Seen such online content',
                  'QPOSTFRQ':'Frequency of such online content',
                  'Q31b':'Engaged with such online content',
                  'QCOVSELF':'Vaccine Intent for self (Pre)', 
                  'QPOSTCOVSELF':'Vaccine Intent for self (Post)',
                  'QCOVOTH':'Vaccine Intent for others (Pre)', 
                  'QPOSTCOVOTH':'Vaccine Intent for others (Post)', 
                  'imageseen':'Group'}
             
    def expand_socc(code):
        names = ['Facebook', 'Twitter', 'YouTube', 'WhatsApp', 'Instagram', 'Pinterest', 'LinkedIN', 'Other', 'None of these']
        out = {}
        for k in code:
             for i in range(len(names)): out['%s%i'%(k, i+1)] = '%s %s'%(names[i], code[k])
        return out
    
    def demo_map(code):
        fwd, bwd = {}, {}
        for key in code:
            fwd[key] = dict(zip(code[key].values(), range(1, len(code[key])+1)))
            bwd[key] = dict(zip(range(1, len(code[key])+1), code[key].values()))
        return fwd, bwd
    
    def expand_imgc(code, num=5):
        out = {}
        for i in range(num):
            for c in code:
                out['%s%i'%(c, i+1)] = 'Image %i:%s'%(i+1, code[c])
        return out
             
    def expand_code(code):
        new = {}
        for key in code:
            new[key] = {}
            for k, v in code[key].items():
                for i in k: new[key][i] = v
        return new
    
    metrics_img = expand_imgc(metrics_img)
    social_atts = expand_socc(social_atts)
    var_fwd, var_bwd = demo_map(var_encoding)
    var_encoding = expand_code(var_encoding)    
    
    atts = list(metrics_any.keys())+list(metrics_knl.keys())+list(metrics_cov.keys())+list(metrics_vci.keys())+list(metrics_aff.keys())
    atts += list(trust[country].keys())+list(reasons.keys())+list(metrics_img.keys())+list(social_atts.keys())
    atts += list(other_atts.keys())+list(demo[country].keys())
    
    def recode_treatment(x): return int('ANTI' in x)
            
    def recode_bools(x): return int('NO TO:' not in x)
        
    def recode_likert(x, inverse=False):
        if inverse: m = {'Strongly agree': -2, 'Tend to agree': -1, 'Tend to disagree': 1, 'Strongly disagree': 2, 'Do not know': 0}
        else: m = {'Strongly agree': 2, 'Tend to agree': 1, 'Tend to disagree': -1, 'Strongly disagree': -2, 'Do not know': 0}
        return m[x]
    
    def recode_likert_num(x, inverse=False):
        if inverse: m = [-2,-1,0,1,2,0]
        else: m = [2,1,0,-1,-2,0]
        return m[x-1]
    
    def recode_age(x):
        if x>118: x = 118
        return (x-18)/100
    
    if group is None:
        idx = df['country']==country
        if country=='UK': idx = idx & ((df['imageseen']=='PRO UK')|(df['imageseen']=='ANTI UK')) #Country field is unreliable, has a bug
        elif country=='USA': idx = idx & ((df['imageseen']=='PRO US')|(df['imageseen']=='ANTI US'))
    else:
        if country=='UK': idx = df['imageseen']==group+' UK'
        elif country=='USA': idx = df['imageseen']==group+' US'    
    
    df_new = df.loc[idx,atts]
    dd_new = {}
    
    for key in metrics_any:
        df_new[key] = df_new[key].apply(recode_bools)
        df_new.rename(columns={key:'Know anyone:%s'%metrics_any[key]}, inplace=True)
        dd_new['Know anyone:%s'%metrics_any[key]] = {1:'Checked', 0:'Unchecked'}
    for key in metrics_knl:
        df_new[key] = df_new[key].apply(recode_likert)
        df_new.rename(columns={key:'COVID-19 Knowledge:%s'%metrics_knl[key]}, inplace=True)
        dd_new['COVID-19 Knowledge:%s'%metrics_knl[key]] = {2:'Strongly agree',1:'Tend to agree',0:'Do not know',-1:'Tend to disagree',-2:'Strongly disagree'}
    for key in metrics_cov:
        df_new[key] = df_new[key].apply(recode_likert)
        df_new.rename(columns={key:'COVID-19 VCI:%s'%metrics_cov[key]}, inplace=True)
        dd_new['COVID-19 VCI:%s'%metrics_cov[key]] = {2:'Strongly agree',1:'Tend to agree',0:'Do not know',-1:'Tend to disagree',-2:'Strongly disagree'}
    for key in metrics_vci:
        df_new[key] = df_new[key].apply(recode_likert)
        df_new.rename(columns={key:'General VCI:%s'%metrics_vci[key]}, inplace=True)
        dd_new['General VCI:%s'%metrics_vci[key]] = {2:'Strongly agree',1:'Tend to agree',0:'Do not know',-1:'Tend to disagree',-2:'Strongly disagree'}
    for key in metrics_aff:
        df_new[key] = df_new[key].apply(recode_likert)
        df_new.rename(columns={key:'COVID-19 Impact:%s'%metrics_aff[key]}, inplace=True)
        dd_new['COVID-19 Impact:%s'%metrics_aff[key]] = {2:'Strongly agree',1:'Tend to agree',0:'Do not know',-1:'Tend to disagree',-2:'Strongly disagree'}
             
    for key in trust[country]:
        df_new[key] = df_new[key].apply(recode_bools)
        df_new.rename(columns={key:'Trust:%s'%trust[country][key]}, inplace=True)
        dd_new['Trust:%s'%trust[country][key]] = {1:'Checked', 0:'Unchecked'}
    for key in reasons:
        df_new[key] = df_new[key].apply(recode_bools)
        df_new.rename(columns={key:'Reason:%s'%reasons[key]}, inplace=True)
        dd_new['Reason:%s'%reasons[key]] = {1:'Checked', 0:'Unchecked'}
    for key in social_atts:
        df_new[key] = df_new[key].apply(recode_bools)
        df_new.rename(columns={key:'Social:%s'%social_atts[key]}, inplace=True)
        dd_new['Social:%s'%social_atts[key]] = {1:'Checked', 0:'Unchecked'}
    
    for key in metrics_img:
        df_new.replace({key: dd['dict'][key]}, inplace=True)
        df_new[key] = df_new[key].apply(recode_likert_num)
        df_new.rename(columns={key:metrics_img[key]}, inplace=True)
        dd_new[metrics_img[key]] = {2:'Strongly agree',1:'Tend to agree',0:'Do not know',-1:'Tend to disagree',-2:'Strongly disagree'}

        
    df_new.replace({att: dd['dict'][att] for att in other_atts if att!='imageseen'}, inplace=True)
    for att in other_atts:
        df_new.rename(columns={att:other_atts[att]}, inplace=True)
        if att!='imageseen': dd_new[other_atts[att]] = dict(zip(dd['dict'][att].values(), dd['dict'][att].keys()))
    
    df_new.replace({key: dd['dict'][key] for key in demo[country] if key not in ['agerecode', 'DGEOUK', 'DGEOUS']}, inplace=True)
    df_new.rename(columns=demo[country], inplace=True)
    df_new.replace(var_encoding, inplace=True)
    df_new.replace(var_fwd, inplace=True)
    for att in demo[country]:
        if demo[country][att] in var_fwd: dd_new[demo[country][att].split('_')[0]] = var_bwd[demo[country][att]]
        else:
            df_new.replace({demo[country][att]: dd['dict'][att]}, inplace=True)
            dd_new[demo[country][att]] = {b: a for (a, b) in dd['dict'][att].items()}
    df_new['Treatment'] = df_new['Group'].apply(recode_treatment)
    del df_new['Group']
    dd_new['Treatment'] = {0: 'Control', 1:'Treatment'}
    df_new.rename(columns={i:i.split('_')[0] for i in list(df_new)}, inplace=True)
    if save:
        df_new.to_csv('%s.csv'%save)
        import pickle
        with open('%s.pkl'%save, 'wb') as fp: pickle.dump(dd_new, fp)
    return df_new, dd_new

def import_transformed_data(filepath=''):
    '''Reads the transformed survey data.
    See questions and field names in /doc/orb_questionnaire.pdf, and refer to recoding in ``transform_data()``

    Args:
        filepath (:obj:`str`): filepath to read the processed data (without the .csv/.pkl suffix)

    Returns:
        A size-2 :obj:`tuple` containing
        * A :obj:`pd.DataFrame` containing field names as columns and record as rows of the transformed data
        * A :obj:`dict` of field names (:obj:`str`) mapped to a :obj:`dict` of recoded-values (:obj:`int`) mapped to value-names (:obj:`str`)
    '''
    import pandas as pd
    import pickle
    df = pd.read_csv('%s.csv'%filepath, index_col=0)
    with open('%s.pkl'%filepath, 'rb') as fp: dd = pickle.load(fp)
    return df, dd

def get_socdem_counts(df, dd, by='Treatment'):
    '''Returns counts of different socio-demographics broken down by a variable of interest.

    Args:
        df (:obj:`pd.DataFrame`): contains transformed data (see return value of ``transform_data()``, ``import_transformed_data()``)
        dd (:obj:`dict`): contains data dictionary for transformed data (see return value of ``transform_data()``, ``import_transformed_data()``)
        by (:obj:`str`): variable of interest; default='Treatment' (returns distribution of demographics across the 2 experiment groups)

    Returns:
        A :obj:`pd.DataFrame` with 2-level index whose outer index corresponds to soc-demo name, inner index to soc-demo value, and columns correspond to % and counts across categories of variable of interest
    '''
    import pandas as pd
    atts = ['Age', 'Gender', 'Education', 'Employment', 'Religion', 'Political', 'Ethnicity', 'Income', 'Social media usage']
    out = []
    for idx, d in df.groupby(by):
        out.append({})
        for att in atts:
            tmp = d[att].value_counts().loc[list(dd[att].keys())]
            tmp.index = dd[att].values()
            tmp.name = '%s (N)'%dd[by][idx]
            tmp_perc = (100*tmp/tmp.sum()).round(1)
            tmp_perc.name = '%s (%%)'%dd[by][idx]
            out[-1][att] = pd.concat([tmp, tmp_perc], axis=1)
        out[-1] = pd.concat(out[-1], axis=0)
    out = pd.concat(out, axis=1)
    return out

def count_attribute(df, att, by_att=None, norm=False, where=None, dd={}, plot=False, att_lab='', by_att_lab='', title='', dpi=90):
    '''Returns counts of any variable of interest, possibly conditioned on a second variable.

    Args:
        df (:obj:`pd.DataFrame`): contains transformed data (see return value of ``transform_data()``, ``import_transformed_data()``)
        att (:obj:`str`): primary variable of interest
        by_att (:obj:`str`): secondary variable of interest to condition counts of the first one on; default=`None`
        norm (:obj:`bool`): whether to normalise the counts to indicate Pr(att); if by_att is not `None` then counts are normalized such that summing Pr(att|by_att) over by_att gives 1
        where (:obj:`list` of size-2 :obj:`tuple` of (:obj:`str`, :obj:`int`)): extra variables to subset the samples on where the tuple encodes a (variable-name, value) pair; default=[]
        dd (:obj:`dict`): contains data dictionary for transformed data (see return value of ``transform_data()``, ``import_transformed_data()``) for sorting counts by given variable-ordering; default={}
        plot (:obj: `bool`): whether to plot the counts; default=`False`
        att_lab (:obj:`str`): if plotting, label for y-axis (primary variable); default=`''`
        by_att_lab (:obj:`str`): if plotting, label for legend (secondary variable); default=`''`
        title (:obj:`str`): if plotting, plot title; default=`''`
        dpi (:obj:`int`): if plotting, dpi for figure; default=90

    Returns:
        A :obj:`pd.DataFrame`/:obj:`pd.Series` whose index corresponds to att and columns to by_att
    '''
    if where is not None:
        if not isinstance(where, list): where = [where]
        for w in where:
            if w[1] is None: df = df[df[w[0]].isnull()]
            else: df = df[df[w[0]]==w[1]]
    if by_att is None: counts = df[att].value_counts()
    else:
        from pandas import concat
        groups = df[[att, by_att]].groupby(by_att)
        names = list()
        counts = list()
        for name, group in groups:
            names.append(name)
            counts.append(group[att].value_counts())
        counts = concat(counts, axis=1, keys=names, sort=True)
        if dd:
            counts = counts[dd[by_att].keys()]
            counts.rename(columns=dd[by_att], inplace=True)
    counts.fillna(0, inplace=True)
    if norm: counts = counts/counts.values.sum(0)
    if dd:
        counts = counts.loc[dd[att].keys()]
        counts.rename(index=dd[att], inplace=True)
    if plot:
        import matplotlib.pyplot as plt
        from seaborn import countplot
        plt.figure(dpi=dpi)
        order, hue_order = None, None
        if dd:
            if by_att is not None:
                df = df[[att,by_att]]
                df = df.replace({att: dd[att], by_att: dd[by_att]})
                hue_order = dd[by_att].values()
            else:
                df = df[[att]]
                df = df.replace({att: dd[att]})
            order = dd[att].values()
        if by_att is None: countplot(y=att, data=df, order=order)
        else: countplot(y=att, hue=by_att, data=df, order=order, hue_order=hue_order)
        plt.gca().set_xlabel('Count')
        if att_lab: plt.gca().set_ylabel(att_lab)
        if by_att_lab: plt.gca().get_legend().set_title(by_att_lab)
        if not title and where is not None: title = ', '.join([str(w[0])+' = '+str(w[1]) for w in where])
        plt.title(title)
        plt.show()        
    return counts