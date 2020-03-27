import pandas as pd
import plotly.graph_objects as go
import factiva_common.taxonomy as tx


def industries_visual_hierarchy() -> pd.DataFrame:
    ret_ind = tx.industries_hierarchy()
    ret_ind['name'] = '(' + ret_ind['ind_fcode'] + ') ' + ret_ind['name']
    return ret_ind


def regions_visual_hierarchy() -> pd.DataFrame:
    ret_reg = tx.regions_hierarchy()
    ret_reg['name'] = '(' + ret_reg['reg_fcode'] + ') ' + ret_reg['name']
    return ret_reg


def publication_timeline(article_df):
    daily_freqs = article_df.groupby('published_at_date', as_index=True).agg({'an': 'count'}).rename(columns={'an': 'daily_volume'}).sort_values(by='published_at_date', ascending=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=daily_freqs.index, 
                              y=daily_freqs['daily_volume'], 
                              mode = 'lines+markers',
                              name='Total Activity'))

    fig1.update_layout(title="Article volume by day", 
                       yaxis={'tickformat': ',3.0', 'showgrid': True, 'zeroline': True, 'rangemode': 'tozero'},
                       xaxis={'showgrid': False, 'zeroline': True},
                       yaxis_title='Article Volume', 
                       xaxis_title='Day')
    fig1.show()
