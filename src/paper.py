from .utils import read_table, organize_df, collapse_df, subset_df, unstack_df, combine_dfs, plot_stats, plot_image_perceptions

def get_tables(datadir='./out', combine_self_others='socdem'):
    import os
    files = os.listdir(datadir)
    table = {}
    if isinstance(combine_self_others, str): combine_self_others = [combine_self_others]
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
    for analysis in combine_self_others:
        for country in table:
            table[country][analysis] = {'stats':{}}
            for kind in ['logOR', 'OR']: table[country][analysis]['stats'][kind] = combine_dfs(table[country]['%s self'%analysis]['stats'][kind], table[country]['%s others'%analysis]['stats'][kind], lsuffix='(self)', rsuffix='(others)', multi=False, axis=0)
            table[country][analysis]['diags'] = combine_dfs(table[country]['%s self'%analysis]['diags'], table[country]['%s others'%analysis]['diags'], lsuffix='(self)', rsuffix='(others)', multi=False, axis=0)
    return table

def causal_effects(tables, plot=True, save='', fmt='pdf'):
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
        if save: save_ate, save_cate = save+'ate', save+'cate'
        else: save_ate, save_cate = save, save
        for effect in data:
            for country in ['UK', 'US']:
                for kind in ['self', 'others']:
                    data[effect].append(tables[country]['impact causal %s'%kind]['stats'][effect].loc['Treatment-Control']*100)
        plot_stats(data['ATE'], oddsratio=False, #title='$\\Delta(y)$', 
                      subtitle=['UK self', 'UK others', 'US self', 'US others'], #xlabel='% Change between Treatment and Control', 
                      bars=True, highlight=True, factor=0.3, title_loc=0.4, save=save_ate)
        plot_stats(data['CATE'], oddsratio=False, #title='$\\Delta_W(y;w)$', 
                      subtitle=['UK self', 'UK others', 'US self', 'US others'], #xlabel='% Change between Treatment and Control', 
                      bars=True, highlight=True, tick_suffix=' [Y]', label_suffix='\n[W]', factor=0.4, title_loc=-0.02, save=save_cate, fmt=fmt)
    return out

def determinants(tables, analysis='socdem', subset='', kind='logOR', factor=0.3, hspace=12, highlight=False, plot=True, save='', fmt='pdf'):
    stats = {}
    diags = {}
    if analysis=='socdem':
        ordering = ['Treatment-Control (self)', 'Treatment-Control (others)']
        subtitle = ['$\\Delta_S$ Self', '$\\Delta_O$ Others']
        label_text = '  $N$'
    else:
        ordering = ['Treatment', 'Control', 'Treatment-Control']
        subtitle = ['Treatment', 'Control', '$\\Delta$']
        label_text = '$(N_T, N_C, N)$'
    for country in ['UK', 'US']:
        stats[country] = organize_df(tables[country][analysis]['stats'][kind], unstack=True, by_first=True)[ordering]
        diags[country] = collapse_df(tables[country][analysis]['diags'][['ESS', 'Rhat']].describe().loc[['min', 'max']].T, fmt=['i', '.2f'])
    out = {'stats': combine_dfs(stats['UK'], stats['US'], lsuffix='UK', rsuffix='US', axis=1, atts=subset),
           'diags': combine_dfs(diags['UK'], diags['US'], lsuffix='UK', rsuffix='US', axis=1)}
    if plot:
        xlab = {'OR':'Odds Ratio', 'logOR':'Log Odds Ratio'}
        orlog = {'OR':True, 'logOR':False}
        title = {'socdem': 'Socio-demographic Determinants (Self) in',
                 'socdem self': 'Socio-demographic Determinants (Self) in', 
                 'socdem others': 'Socio-demographic Determinants (Others) in',
                 'social': 'Social Media Usage in',
                 'trust': 'Sources of COVID-19 Info Trusted in'}
        title_loc = {'socdem': (0.04, 0.04),
                     'socdem self': (0.04, 0.04),
                     'socdem others': (0.04, 0.04),
                     'social': (0.2, 0.1),
                     'trust': (0.08, 0.04)}
        countries = {'UK': 'a', 'US': 'b'}
        fig, ax, ax_outer = None, None, None
        demos = not(bool(subset))
        if save: save = ['', save+'hte_%s'%'_'.join(analysis.split())]
        else: save = ['', '']
        for country, idx in zip(countries, range(2)):
            df = tables[country][analysis]['stats'][kind]
            fig, ax, ax_outer = plot_stats([df.loc[i] for i in ordering], fig=fig, ax=ax, ax_outer=ax_outer, fignum=2, figidx=idx, demos=demos, oddsratio=orlog[kind], title='%s %s'%(countries[country], country), highlight=highlight, identical_counts=analysis=='socdem',
                subtitle=subtitle, label_text=label_text, hspace=hspace, stack_h=demos, widespace=(3-1.75*demos-0.3*(analysis=='socdem')), ylabel=demos and idx==1, factor=factor, title_loc=title_loc[analysis][idx], align_labels=True, capitalize=True, show=idx==2, save=save[idx], fmt=fmt)
    return out

def image_perceptions(tables, plot=True, save='', fmt='pdf'):
    data = []
    for group in ['Treatment', 'Control']:
        for country in ['UK', 'US']: data.append(tables[country]['image perception']['means'].loc[group])
    if save: save = save+'image_perceptions'
    if plot: plot_image_perceptions(data, ['a UK treatment', 'b US treatment', 'c UK control', 'd US control'], save=save, fmt=fmt)
    out_T = combine_dfs(unstack_df(data[0], by_first=True), unstack_df(data[1], by_first=True), lsuffix='UK', rsuffix='US')
    out_C = combine_dfs(unstack_df(data[2], by_first=True), unstack_df(data[3], by_first=True), lsuffix='UK', rsuffix='US')
    out = 100*combine_dfs(out_T, out_C, lsuffix='Treatment', rsuffix='Control')
    return out

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

def similar_content(tables, subset=True, plot=True, save='', fmt='pdf'):
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
        if save: save = save+'similar_content'
        for country in ['UK', 'US']:
            tmp = pd.DataFrame.droplevel(subset_df(unstack_df(tables[country]['filterbubble']['stats'], by_first=True), atts)*100, 1)
            for group in ['Pre-exposure', 'Post-exposure']: data.append(tmp[group])
        plot_stats(data, oddsratio=False, title='Difference in vaccine intents conditioned on\nhaving previously seen COVID-19 (mis)information', subtitle=['UK (Pre-exposure)', 'UK (Post-exposure)', 'US (Pre-exposure)', 'US (Post-exposure)'], factor=0.4, subtitlesize=16, titlesize=20, title_loc=0.15, bars=True, save=save, fmt=fmt)
    return out