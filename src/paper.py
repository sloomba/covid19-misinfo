from .utils import read_table, organize_df, collapse_df, subset_df, unstack_df, combine_dfs, plot_stats, plot_image_perceptions

def get_tables(datadir='./out'):
    import os
    files = os.listdir(datadir)
    table = {}
    for name in files:
        if name[-4:]=='.csv':
            tokens = name[:-4].split('_')
            country = tokens[1].upper()
            if country not in table: table[country] = {}
            for s in ['stats', 'diags', 'means']:
                if s in tokens:
                    analysis = [x.strip() for x in ' '.join(tokens[2:]).split(s)]
                    if analysis[0] not in table[country]: table[country][analysis[0]] = {}
                    if analysis[1]:
                        if s not in table[country][analysis[0]]: table[country][analysis[0]][s] = {}
                        table[country][analysis[0]][s][analysis[1]] = read_table(name, datadir)
                    else: table[country][analysis[0]][s] = read_table(name, datadir)
                    break
    return table

def causal_effects(tables, plot=True):
    stats = {}
    diags = {}
    ordering = ['Baseline', 'Treatment-Control', 'Treatment', 'Control']
    for country in ['UK', 'US']:
        stat, diag = {}, {}
        for kind in ['self', 'others']:
            stat[kind] = {}
            for effect in ['ATE', 'CATE']: stat[kind][effect] = organize_df(tables[country]['impact causal %s'%kind]['stats'][effect], perc=True, atts=ordering)
            diag[kind] = collapse_df(tables[country]['impact causal %s'%kind]['diags'][['ESS', 'Rhat']].describe().loc[['min', 'max']].T, fmt=['i', '.2f'])
        stats[country] = {effect: combine_dfs(stat['self'][effect], stat['others'][effect], lsuffix='Self', rsuffix='Others') for effect in ['ATE', 'CATE']}
        diags[country] = combine_dfs(diag['self'], diag['others'], lsuffix='Self', rsuffix='Others', axis=0)
    out = {'stats': {effect: combine_dfs(stats['UK'][effect], stats['US'][effect], lsuffix='UK', rsuffix='US') for effect in ['ATE', 'CATE']}, 
           'diags': combine_dfs(diags['UK'], diags['US'], lsuffix='UK', rsuffix='US')}
    if plot:
        data = {'ATE':[], 'CATE':[]}
        for effect in data:
            for country in ['UK', 'US']:
                for kind in ['self', 'others']:
                    data[effect].append(tables[country]['impact causal %s'%kind]['stats'][effect].loc['Treatment-Control']*100)
        plot_stats(data['ATE'], oddsratio=False, title='$\\Delta(y)$', 
                      subtitle=['UK Self', 'UK Others', 'US Self', 'US Others'],
                      xlabel='% Change between Treatment and Control', bars=True, highlight=True, factor=0.25, title_loc=0.4)
        plot_stats(data['CATE'], oddsratio=False, title='$\\Delta_W(y;w)$', 
                      subtitle=['UK Self', 'UK Others', 'US Self', 'US Others'],
                      xlabel='% Change between Treatment and Control', bars=True, highlight=True, tick_suffix=' [Y]', label_suffix='\n[W]', factor=0.3, labelsize=7, title_loc=-0.02)
    return out

def determinants(tables, analysis='socdem self', subset='', kind='logOR', plot=True):
    stats = {}
    diags = {}
    ordering = ['Treatment', 'Control', 'Treatment-Control']
    for country in ['UK', 'US']:
        stats[country] = organize_df(tables[country][analysis]['stats'][kind], unstack=True, by_first=True)[ordering]
        diags[country] = collapse_df(tables[country][analysis]['diags'][['ESS', 'Rhat']].describe().loc[['min', 'max']].T, fmt=['i', '.2f'])
    out = {'stats': combine_dfs(stats['UK'], stats['US'], lsuffix='UK', rsuffix='US', axis=1, atts=subset),
           'diags': combine_dfs(diags['UK'], diags['US'], lsuffix='UK', rsuffix='US', axis=1)}
    if plot:
        xlab = {'OR':'Odds Ratio', 'logOR':'Log Odds Ratio'}
        orlog = {'OR':True, 'logOR':False}
        title = {'socdem self': 'Socio-demographic Determinants in', 
                 'socdem others': 'Socio-demographic Determinants (Others) in',
                 'social': 'Social Media Usage in',
                 'trust': 'Sources of COVID-19 Info Trusted in', 
                 'reason': 'Reasons for COVID-19 Vaccine Hesitancy in'}
        title_loc = {'socdem self': -0.04, 
                     'socdem others': -0.04, 
                     'social': 0.2,
                     'trust': 0.01, 
                     'reason': 0.08}
        countries = {'UK': 'A', 'US': 'B'}
        for country in countries:
            df = tables[country][analysis]['stats'][kind]
            plot_stats([df.loc['Treatment'], df.loc['Control'], df.loc['Treatment-Control']], demos=not(bool(subset)), oddsratio=orlog[kind], title='(%s) %s %s'%(countries[country], title[analysis], country), 
                subtitle=['Treatment', 'Control', '$\\Delta$'], label_text='$(N_T, N_C, N)$', xlabel=xlab[kind], ylabel=not(bool(subset)), factor=0.4, titlesize=22, subtitlesize=16, labelsize=12, title_loc=title_loc[analysis], align_labels=True)
    return out

def image_perceptions(tables):
    data = []
    for group in ['Treatment', 'Control']:
        for country in ['UK', 'US']: data.append(tables[country]['image perception']['means'].loc[group])
    plot_image_perceptions(data, ['(A) UK Treatment', '(B) US Treatment', '(C) UK Control', '(D) US Control'], legend_loc=(0.07, -0.05))

def image_impact(tables, kind='logOR'):
    stats = {}
    diags = {}
    metrics = ['Makes less inclined to vaccinate', 'Agree with', 'Found trustworthy', 'Likely to fact-check', 'Likely to share']
    params = metrics + ['Image %i'%(i+1) for i in range(5)]
    def set_index(df, index):
        df['index'] = index
        df.set_index('index', verify_integrity=True, inplace=True)
        del df.index.name
    for group in ['C', 'T']:
        stats[group] = combine_dfs(collapse_df(tables['UK']['image impact %s'%group]['stats'][kind]),
            collapse_df(tables['US']['image impact %s'%group]['stats'][kind]), lsuffix='UK', rsuffix='US', axis=1)
        set_index(stats[group], params)
        diags[group] = combine_dfs(collapse_df(tables['UK']['image impact %s'%group]['diags'][['ESS', 'Rhat']].describe().loc[['min', 'max']].T, fmt=['i', '.2f']),
            collapse_df(tables['US']['image impact %s'%group]['diags'][['ESS', 'Rhat']].describe().loc[['min', 'max']].T, fmt=['i', '.2f']), lsuffix='UK', rsuffix='US', axis=1)
    return {'stats': combine_dfs(stats['T'], stats['C'], lsuffix='Treatment', rsuffix='Control', axis=1),
            'diags': combine_dfs(diags['T'], diags['C'], lsuffix='Treatment', rsuffix='Control', axis=0)}

def similar_content(tables, subset=True, plot=True):
    import pandas as pd
    stats = {}
    diags = {}
    if subset:
        atts = [('Pre-exposure', 'Control', 'Seen-Not seen'), ('Pre-exposure', 'Treatment', 'Seen-Not seen'), ('Pre-exposure', 'Treatment-Control', 'Seen'),
                ('Post-exposure', 'Control', 'Seen-Not seen'), ('Post-exposure', 'Treatment', 'Seen-Not seen'), ('Post-exposure', 'Treatment-Control', 'Seen')]
        atts = [('Control', 'Seen-Not seen'), ('Treatment', 'Seen-Not seen'), ('Treatment-Control', 'Seen')]
    else: atts = []
    for country in ['UK', 'US']:
        stats[country] = organize_df(tables[country]['filterbubble']['stats'], perc=True, unstack=True, by_first=True, atts=atts)
        diags[country] = collapse_df(tables[country]['filterbubble']['diags'][['ESS', 'Rhat']].describe().loc[['min', 'max']].T, fmt=['i', '.2f'])
    out = {'stats': combine_dfs(stats['UK'], stats['US'], lsuffix='UK', rsuffix='US'),
           'diags': combine_dfs(diags['UK'], diags['US'], lsuffix='UK', rsuffix='US')}
    if subset: out['stats'] = pd.DataFrame.droplevel(out['stats'], 1)
    if plot:
        data = []
        for country in ['UK', 'US']:
            tmp = pd.DataFrame.droplevel(subset_df(unstack_df(tables[country]['filterbubble']['stats'], by_first=True), atts)*100, 1)
            for group in ['Pre-exposure', 'Post-exposure']: data.append(tmp[group])
        plot_stats(data, oddsratio=False, title='Difference in vaccine intents conditioned on\nhaving previously seen COVID-19 (mis)information', subtitle=['UK (Pre-exposure)', 'UK (Post-exposure)', 'US (Pre-exposure)', 'US (Post-exposure)'], factor=0.4, subtitlesize=16, titlesize=20, title_loc=0.15, bars=True)
    return out