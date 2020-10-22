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

def transform_data(df, dd, country='UK', group=None, minimal=True, save=''):
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
    if minimal:
        other_atts = {'QSOCUSE':'Social media usage', 
                      'QPOSTSIM':'Seen such online content',
                      'QCOVSELF':'Vaccine Intent for self (Pre)', 
                      'QPOSTCOVSELF':'Vaccine Intent for self (Post)',
                      'QCOVOTH':'Vaccine Intent for others (Pre)', 
                      'QPOSTCOVOTH':'Vaccine Intent for others (Post)', 
                      'imageseen':'Group'}
    else:
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
    
    if minimal: atts = []
    else: atts = list(metrics_any.keys())+list(metrics_knl.keys())+list(metrics_cov.keys())+list(metrics_vci.keys())+list(metrics_aff.keys())+list(social_atts.keys())
    atts += list(trust[country].keys())+list(reasons.keys())+list(metrics_img.keys())
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
    
    if not minimal:
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
        for key in social_atts:
            df_new[key] = df_new[key].apply(recode_bools)
            df_new.rename(columns={key:'Social:%s'%social_atts[key]}, inplace=True)
            dd_new['Social:%s'%social_atts[key]] = {1:'Checked', 0:'Unchecked'}
             
    for key in trust[country]:
        df_new[key] = df_new[key].apply(recode_bools)
        df_new.rename(columns={key:'Trust:%s'%trust[country][key]}, inplace=True)
        dd_new['Trust:%s'%trust[country][key]] = {1:'Checked', 0:'Unchecked'}
    for key in reasons:
        df_new[key] = df_new[key].apply(recode_bools)
        df_new.rename(columns={key:'Reason:%s'%reasons[key]}, inplace=True)
        dd_new['Reason:%s'%reasons[key]] = {1:'Checked', 0:'Unchecked'}    
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

def stats_impact(fit):
    import numpy as np
    from .bayesoc import Outcome, Model
    import pandas as pd
    m = 2
    k = 4
    def foo(x): return np.diff(np.hstack([0, np.exp(x)/(1+np.exp(x)), 1]))
    df = Model(Outcome()).get_posterior_samples(fit=fit)
    prob = []
    names = ['Yes, definitely', 'Unsure, lean yes', 'Unsure, lean no', 'No, definitely not']
    for i in range(1, m+1):
        alpha = df[['alpha[%i,%i]'%(i, j) for j in range(1, k)]].values
        p = []
        for a in alpha: p.append(foo(a))
        p = np.vstack(p)
        prob.append(pd.DataFrame(p, columns=names))
    prob.append(prob[1]-prob[0])
    groups = ['Pre Exposure', 'Post Exposure', 'Post-Pre']
    out = pd.concat({groups[i]: prob[i].describe(percentiles=[0.025, 0.975]).T[['mean', '2.5%', '97.5%']] for i in range(len(groups))})
    return out

def stats_impact_causal(fit):
    import numpy as np
    from .bayesoc import Outcome, Model
    import pandas as pd
    m = 2
    k = 4
    def foo(x): return np.diff(np.hstack([0, np.exp(x)/(1+np.exp(x)), 1]))
    df = Model(Outcome()).get_posterior_samples(fit=fit)
    prob = []
    dfs = [[], [], []]
    names = ['Yes, definitely', 'Unsure, lean yes', 'Unsure, lean no', 'No, definitely not']
    for i in range(1, m+1):        
        beta = np.hstack([np.zeros((df.shape[0],1)), df[['beta[%i]'%i]].values*df[['delta[%i,%i]'%(i, j) for j in range(1, k)]].values.cumsum(axis=1)])
        alpha = df[['alpha[%i,%i]'%(i, j) for j in range(1, k)]].values
        p = []
        for (a, b) in zip(alpha, beta): p.append(np.array([foo(a-b_) for b_ in b]))
        prob.append(np.dstack(p))
        for j in range(k):
            dfs[i-1].append(pd.DataFrame(prob[-1][j].T, columns=names).describe(percentiles=[0.025, 0.975]).T[['mean', '2.5%', '97.5%']])
    diff = prob[1]-prob[0]
    for i in range(k):
        dfs[-1].append(pd.DataFrame(diff[i].T, columns=names).describe(percentiles=[0.025, 0.975]).T[['mean', '2.5%', '97.5%']])
    groups = ['Control', 'Treatment', 'Treatment-Control']
    out = pd.concat({groups[i]: pd.concat({names[j]: dfs[i][j] for j in range(len(names))}) for i in range(len(groups))})
    return out

def plot_stats(df1, df2=None, demos=False, oddsratio=True, title='', title_l='', title_r='', xlab='', xlab_l='', xlab_r='', tick_suffix='', label_suffix='', ylabel=True, factor=0.3, signsize=10, ticksize=10, labelsize=12, titlesize=14, hspace=0.2, wspace=0.05, align_labels=False, title_loc=0.0):
    import matplotlib.pyplot as plt
    import numpy as np
    dem = ['Age', 'Gender', 'Education', 'Employment', 'Religion', 'Political', 'Ethnicity', 'Income']
    atts = list(df1.index)
    if not isinstance(atts[0], tuple):
        from pandas import concat
        df1 = concat({'tmp': df1})
        if df2 is not None: df2 = concat({'tmp': df2})
        atts = list(df1.index)
        ylabel = False
    if not demos: atts = [i for i in atts if i[0] not in dem]
    att_cat = {}
    for att in atts:
        if ':' in att[0]:
            key, val = tuple(att[0].split(':'))
            if key in att_cat:
                att_cat[key]['idx'].append(att)
                att_cat[key]['val'].append(val+tick_suffix)
            else: att_cat[key] = {'idx': [att], 'val': [val+tick_suffix]}
        else:
            if att[0] in att_cat:
                att_cat[att[0]]['idx'].append(att)
                att_cat[att[0]]['val'].append(att[1]+tick_suffix)
            else: att_cat[att[0]] = {'idx': [att], 'val': [att[1]+tick_suffix]}
                
    rows = len(att_cat)
    rows_per = [len(att_cat[k]['idx']) for k in att_cat]
    cols = 2-int(df2 is None)
    fig, ax = plt.subplots(nrows=rows, ncols=cols, dpi=180, sharex='col', figsize=(6, factor*sum(rows_per)),
                           gridspec_kw={'height_ratios': rows_per}, constrained_layout=not(bool(xlab)))
    if len(ax.shape)==1:
        if df2 is None: ax = ax[:,np.newaxis]
        else: ax = ax[np.newaxis,:]
    names = list(att_cat.keys())
    
    def plot_bars(ax, tmp, ticks=[], right=False, base=False):
        num = tmp.shape[0]
        if base:
            ax.barh(y=list(range(num+1, num+2-base, -1))+list(range(num+1-base, 0, -1)), width=tmp['mean'].values, xerr=np.vstack([(tmp['mean']-tmp['2.5%']).values, (tmp['97.5%']-tmp['mean']).values]), color='salmon')
            ax.text(0, num+2-base, 'REFERENCE', size=ticksize)
        else: ax.barh(y=range(num, 0, -1), width=tmp['mean'].values, xerr=np.vstack([(tmp['mean']-tmp['2.5%']).values, (tmp['97.5%']-tmp['mean']).values]), color='salmon')
        ax.axvline(oddsratio, ls='--', color='black')
        for i in range(num):
            lb, ub = tmp['2.5%'][tmp.index[i]], tmp['97.5%'][tmp.index[i]]
            if not (lb<oddsratio<ub):
                if lb<0: ax.text(lb, num-i, '*', size=signsize)
                else: ax.text(ub, num-i, '*', size=signsize)
        if ticks: t = range(1, num+1)
        else: t = []
        ax.yaxis.set_ticklabels(reversed(ticks))
        ax.yaxis.set_ticks(t)
        if right: ax.yaxis.tick_right()
    
    for i in range(rows):
        tmp = df1.loc[att_cat[names[i]]['idx']]
        #if names[i] in dem: plot_bars(ax[i,0], tmp, list(dd[names[i]].values()), base=base[names[i]])
        #else: plot_bars(ax[i,0], tmp, att_cat[names[i]]['val'])
        plot_bars(ax[i,0], tmp, att_cat[names[i]]['val'])
        if ylabel: ax[i,0].set_ylabel(names[i]+label_suffix, fontweight='bold', fontsize=labelsize)
        #else: plot_bars(ax[i], tmp, att_cat[names[i]]['val'])
        if df2 is not None:
            try: c1 = tmp['counts'].values
            except: c1 = []
            tmp = df2.loc[att_cat[names[i]]['idx']]
            try: c2 = tmp['counts'].values
            except: c2 = []
            plot_bars(ax[i,1], tmp, ['%i, %i'%(a, b) for a, b in zip(c1, c2)], right=True)
            #plot_bars(ax[i,1], tmp)
            if i==0:
                if title_l: ax[i,0].set_title(title_l)
                if title_r: ax[i,1].set_title(title_r)
    if align_labels: fig.align_ylabels()
    if title: plt.suptitle(title, fontsize=titlesize, y=1+title_loc)
    if xlab_l and xlab_r:
        ax[i,0].set_xlabel(xlab_l, fontsize=labelsize)
        ax[i,1].set_xlabel(xlab_r, fontsize=labelsize)
    if xlab:
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel(xlab, fontsize=labelsize)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
    #fig.tight_layout()
    plt.show()
    return

def plot_causal_flow(df, df_T, df_C, title='', save=''):
    def plot_sankey(group, dat):
        import plotly.graph_objects as go
        src, tgt, val = [], [], []
        labs = ['Yes, definitely', 'Unsure, lean yes', 'Unsure, lean no', 'No, definitely not']*2
        for i in range(4):
            for j in range(4, 8):
                src.append(i)
                tgt.append(j)
                val.append(df.loc[(group,labs[i],labs[j]), 'mean']*dat.loc[('Pre Exposure',labs[i]), 'mean'])
        fig = go.Figure(data=[go.Sankey( 
        node = dict(pad=15, thickness=40, line=dict(color='salmon', width=0.5), color='salmon',
                    label=['[%i] %s'%(round(100*y), x) for x, y in zip(labs[:4], dat.loc['Pre Exposure', 'mean'])]+['%s [%i]'%(x, round(100*y)) for x, y in zip(labs[4:], dat.loc['Post Exposure', 'mean'])]),
        link = dict(source=src, target=tgt, value=val))])
        fig.update_layout(title_text='%s %s'%(title, group), font_size=20)
        fig.show()
        if save: fig.write_image('%s_%s.png'%(save, group), scale=4)
    plot_sankey('Treatment', df_T)
    plot_sankey('Control', df_C)

def stats_socdem(fit, dd, df, atts=[], group=None, oddsratio=True, title='Trust in Source'):
    import numpy as np
    import pandas as pd
    from .bayesoc import Dim, Outcome, Model
    import matplotlib.pyplot as plt
    cats = ['Age', 'Gender', 'Education', 'Employment', 'Religion', 'Political', 'Ethnicity', 'Income']
    if isinstance(atts, str): atts = [atts]
    for att in atts: cats += [x for x in list(df) if x[:len(att)]==att]
    outs = ['Vaccine Intent for self (Pre)', 'Vaccine Intent for self (Post)', 'Treatment']
    bases = [1]*len(cats) #default reference category for all socio-demographics
    bases[2] = 5 #reference category for education
    bases[7] = 5 #reference category for income
    tmp = Model(Outcome())
    stats = {}
    counts_all = {}
    def foo(x): return np.exp(x)
    for cat, base in zip(cats, bases):
        vals = np.sort(list(dd[cat].keys()))
        if group is None: counts = df[cat].value_counts().loc[vals]
        else: counts = df[df['Treatment']==group][cat].value_counts().loc[vals]
        counts.index = [dd[cat][k] for k in vals]
        counts_all[cat] = counts.iloc[list(range(base-1))+list(range(base, len(vals)))]
        dim = Dim(name=cat)
        stats[cat] = tmp.get_posterior_samples(pars=['beta_%s[%i]'%(dim.name, i+1) for i in range(len(dd[cat]))], contrast='beta_%s[%i]'%(dim.name, base), fit=fit)
        if oddsratio: stats[cat] = stats[cat].apply(foo)
        stats[cat] = stats[cat].describe(percentiles=[0.025, 0.975]).T[['mean', '2.5%', '97.5%']]
        stats[cat].drop('chain', inplace=True)
        stats[cat].index = counts_all[cat].index
    stats = pd.concat(stats)
    counts = pd.concat(counts_all)
    counts.name = 'counts'
    return stats.merge(counts.to_frame(), left_index=True, right_index=True)

def stats_image_perceptions(fit, num_levels=5):
    import numpy as np
    import pandas as pd
    from .bayesoc import Outcome, Model
    def foo(x): return np.diff(np.hstack([0, np.exp(x)/(1+np.exp(x)), 1]))
    tmp = Model(Outcome())
    out = {}
    for i in range(len(fit)):
        out[i+1] = {}
        for m in fit[i]:
            df = tmp.get_posterior_samples(fit=fit[i][m])
            out[i+1][m] = pd.DataFrame(np.array([foo(x) for x in df[['alpha[%i]'%(i+1) for i in range(num_levels-1)]].values]), columns= ['p[%i]'%(j+1) for j in range(num_levels)]).describe(percentiles=[0.025, 0.975]).T[['mean', '2.5%', '97.5%']]
        out[i+1] = pd.concat(out[i+1])
    return pd.concat(out)

def plot_image_perceptions(df, ylab=[], imagewise=False, legend_loc=0.1):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    
    if not isinstance(df, list): df = [df]
    questions = {'Vaccine Intent': 'Raises Vaccine Intent', 'Agreement': 'Agree with', 'Trust': 'Have trust in', 'Fact-check': 'Will fact-check', 'Share': 'Will share'}
    categories = dict(zip(['p[%i]'%(i+1) for i in range(5)], ['Strongly disagree', 'Somewhat disagree', 'Neither', 'Somewhat agree', 'Strongly agree']))
    
    def survey(results, category_names, ax=None):
        """
        Parameters
        ----------
        results : dict
            A mapping from question labels to a list of answers per category.
            It is assumed all lists contain the same number of entries and that
            it matches the length of *category_names*.
        category_names : list of str
            The category labels.
        """
        # Ref: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html
        labels = list(results.keys())
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)
        category_colors = plt.get_cmap('RdYlBu')(np.linspace(0.15, 0.85, data.shape[1]))

        if ax is None: fig, ax = plt.subplots(dpi=90, figsize=(5, 5))
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)
            xcenters = starts + widths / 2
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'dimgray'
            for y, (x, c) in enumerate(zip(xcenters, widths)):
                ax.text(x, y, str(int(100*c)), ha='center', va='center', color=text_color)
        return ax
    
    matplotlib.rc('font', size=18)
    nrows = len(df)
    fig, ax = plt.subplots(nrows=nrows, ncols=5, dpi=90, figsize=(25, 5*nrows))#, constrained_layout=True)
    ax_sup = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #plt.subplots_adjust(hspace=hspace, wspace=wspace)
    if nrows==1: ax = ax[np.newaxis,:]
    if imagewise:
        for p in range(nrows):
            for i in range(5):
                results = {questions[j]: df[p]['mean'][(i+1, j)][list(categories.keys())].values for j in questions}
                results['Raises Vaccine Intent'] = results['Raises Vaccine Intent'][::-1]
                survey(results, categories.values(), ax[p,i])
                if p==0: ax[p,i].set_title('Image %i'%(i+1))
                if i==0 and ylab: ax[p,i].set_ylabel(ylab[p], fontsize=24, fontweight='bold')
    else:
        for p in range(nrows):
            flag = True
            for j, a in zip(questions, ax[p]):
                if j=='Vaccine Intent': results = {'Img %i'%(i+1): df[p]['mean'][(i+1, j)][list(categories.keys())].values[::-1] for i in range(5)}
                else: results = {'Img %i'%(i+1): df[p]['mean'][(i+1, j)][list(categories.keys())].values for i in range(5)}
                survey(results, categories.values(), a)
                if p==0: a.set_title(questions[j])
                if flag and ylab:
                    a.set_ylabel(ylab[p], fontsize=24, fontweight='bold')
                    flag = False
    plt.tight_layout()
    handles, labels = ax[0,0].get_legend_handles_labels()
    ax_sup.legend(handles=handles, labels=labels, ncol=5, bbox_to_anchor=(0, -legend_loc), loc='lower left', fontsize=22.5)
    plt.show()

def stats_image_impact(fit, oddsratio=False, plot=False, num_metrics=5, num_images=5):
    import numpy as np
    import pandas as pd
    from .bayesoc import Outcome, Model
    tmp = Model(Outcome())
    pars = ['beta[%i]'%(i+1) for i in range(num_metrics)]
    if plot: tmp.plot_posterior_pairs(fit=fit, pars=pars)
    pars2 = ['gamma[%i]'%(i+1) for i in range(num_images)]
    df = tmp.get_posterior_samples(pars=pars, fit=fit)
    def foo(x): return np.exp(x)
    if oddsratio: return df[pars].apply(foo).merge(tmp.get_posterior_samples(pars=pars2, fit=fit)[pars2], left_index=True, right_index=True).describe(percentiles=[0.025, 0.975]).T[['mean', '2.5%', '97.5%']]
    else: return df[pars].merge(tmp.get_posterior_samples(pars=pars2, fit=fit)[pars2], left_index=True, right_index=True).describe(percentiles=[0.025, 0.975]).T[['mean', '2.5%', '97.5%']]

def stats_filterbubble(fit, contrast=False):
    import numpy as np
    from .bayesoc import Outcome, Model
    import pandas as pd
    m = 2
    k = 4
    def foo(x): return np.diff(np.hstack([0, np.exp(x)/(1+np.exp(x)), 1]))
    df = Model(Outcome()).get_posterior_samples(fit=fit)
    beta = df[['beta[%i]'%i for i in range(1, k+1)]].values
    alpha = df[['alpha[%i]'%i for i in range(1, m)]].values
    p = []
    for (a, b) in zip(alpha, beta): p.append(np.array([foo(a-b_) for b_ in b]))
    p = np.dstack(p)[:,-1,:]
    if contrast:
        p = (p-p[0])[1:]
        names = ['Unsure, lean yes', 'Unsure, lean no', 'No, definitely not']
    else: names = ['Yes, definitely', 'Unsure, lean yes', 'Unsure, lean no', 'No, definitely not']
    out = pd.concat([pd.DataFrame(p[i].T, columns=['Yes']).describe(percentiles=[0.025, 0.975]).T[['mean', '2.5%', '97.5%']] for i in range(len(names))])
    out.index = names
    return out

def combine_dfs(df_l, df_r, lsuffix='(1)', rsuffix='(2)', collapse=True, perc=False, atts=[]):
    if collapse:
        def foo(df):
            import pandas as pd
            out = []
            for x, (lb, ub) in zip(df['mean'].values, zip(df['2.5%'].values, df['97.5%'].values)):
                if perc: out.append('%.1f (%.1f, %.1f)'%(100*x, 100*lb, 100*ub))
                else: out.append('%.2f (%.2f, %.2f)'%(x, lb, ub))
            out = pd.DataFrame(out, index=df.index, columns=['Value'])
            return out
        df_l = foo(df_l)
        df_r = foo(df_r)
    df = df_l.join(df_r, lsuffix=' '+lsuffix, rsuffix=' '+rsuffix, how='outer')
    #df.fillna('-', inplace=True)
    if atts:
        to_use = []
        for att in atts: to_use += [a[0] for a in df.index if a[0][:len(att)]==att]
        df = df.loc[to_use,:]
    return df

def unstack_df(df, perc=False):
    def foo(df):
        import pandas as pd
        out = []
        for x, (lb, ub) in zip(df['mean'].values, zip(df['2.5%'].values, df['97.5%'].values)):
            if perc: out.append('%.1f (%.1f, %.1f)'%(100*x, 100*lb, 100*ub))
            else: out.append('%.2f (%.2f, %.2f)'%(x, lb, ub))
        out = pd.Series(out, index=df.index)
        return out
    return foo(df).unstack()