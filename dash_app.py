import datetime
import sqlite3
from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.express as px
import polars as pl
from dash import Dash, Input, Output, State, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate

import dash_dataframe_table

parent_dir = Path().absolute().stem


def make_headline_info_div(headline, tooltip_text, id):
    return html.Div(
        children=[
            html.H3(headline, style={'display': 'inline-block'}),
            html.I(
                id=id,
                className='bi bi-info-circle-fill me-2',
                style={'display': 'inline-block', 'padding': '.5em'},
            ),
            dbc.Tooltip(
                dcc.Markdown(tooltip_text),
                target=id,
                trigger='legacy',
            ),
        ],
    )


def get_date(wordle_num):
    return (
        datetime.datetime(2021, 6, 19) + datetime.timedelta(days=int(wordle_num))
    ).strftime('%Y-%m-%d')


def num_to_color(pattern: str):
    return pattern.replace('0', '⬛').replace('1', '🟨').replace('2', '🟩')


with sqlite3.connect('wordle.db') as con:
    wordle_solutions = pl.read_database(
        'select distinct(word), wordle_id from solutions where is_solution = 1 order by word asc',
        connection=con,
    )
    options_wordle = [
        {'label': label, 'value': f'{label}|{value}'}
        for label, value in wordle_solutions.select('word', 'wordle_id').iter_rows()
    ]

app = Dash(
    __name__,
    url_base_pathname=f'/dash/{parent_dir}/',
    external_stylesheets=[dbc.themes.YETI, dbc.icons.BOOTSTRAP],
    title='Wordle Explorer',
    meta_tags=[
        {'name': 'viewport', 'content': 'width=device-width, initial-scale=1'},
    ],
    suppress_callback_exceptions=True,
)


def add_header_tooltip(header):
    simple_header = header.replace(' ', '_').lower()
    #  print(simple_header, 'simple header')
    tooltip_lookup = {
        'total_valid_patterns': 'Number of all possible patterns that could exist based on the allowed guess list.',
        'fraction_found': 'Number of unique patterns found in shared posts divided by total possible patterns.',
        'valid_patterns_found': 'Unique valid patterns found',
        'impossible_pattern_count': 'Impossible pattern counts. *Not* unique impossible patterns but includes the guess count.',
        'norm_score': 'Normalized score - a normalized sum of the frequency of the words that could make the pattern, minus a pentalty terms for impossible patterns',
        'kstatistic': 'Two-sample Kolmogorov-Smirnov test statistic from SciPy comparing the distributions of the *first guess* to the NYT published opener frequencies.',
    }
    tooltip = tooltip_lookup.get(simple_header)
    #  print(f'the tooltip for column {simple_header} found is {tooltip}')
    thetarget = f'span-header-{simple_header}'
    if tooltip:
        return html.Div([
            html.Span(header),
            html.I(
                id=thetarget,
                className='bi bi-info-circle-fill',
                style={
                    'display': 'inline-block',
                    'padding-left': '.5em',
                },
            ),
            dbc.Tooltip(dcc.Markdown(tooltip), target=thetarget, trigger='legacy'),
        ])
    return html.Div(header)


tab1_content = html.Div([
    html.H1(children='Wordle Solution Explorer App', style={'textAlign': 'center'}),
    html.Div(
        id='dropdown-div',
        children=dbc.Select(id='wordle-input', persistence=True, persistence_type='session'),
    ),
    dbc.Container(
        children=[
            html.Div([
                dbc.Row(
                    [
                        dbc.Col([
                            html.Div(id='pattern-div'),
                        ]),
                        dbc.Col(html.Div(id='graph-div')),
                    ],
                ),
                dbc.Row(html.Div(id='weird-guesses')),
                dbc.Row(html.Div(id='table-div')),
            ])
        ],
        # id='my-output',
        style={'text-align': 'center', 'margin': 'auto'},
    ),
])

with open('README.md', 'r') as f:
    tab_about_content = f.read()

with sqlite3.connect('wordle.db') as con:
    df = pl.read_database(
        'select max(wordle_id) wordle_id,count(*) as count from solutions where is_solution = 1 order by 1 asc',
        connection=con,
    )
    df_pct_correct = pl.read_database(
        'select avg(is_solution) is_solution from (select word,is_solution from solutions group by wordle_id having min(rowid) order by rowid);',
        connection=con,
    )['is_solution'].item(0)
    df_last_correct = pl.read_database(
        'select word,is_solution from solutions where wordle_id = (select max(wordle_id) from solutions) order by rowid limit 1;',
        connection=con,
    )['is_solution'].item(0)
    max_wordle = df['wordle_id'].item(0)
    num_solutions = df['count'].item(0)
    max_date = get_date(wordle_num=max_wordle)
    df_bad_norm_scores = pl.read_database(
        """select * from ranked_view where norm_score_rank  <> 1 and is_solution = 1;""",
        connection=con,
    )
    df_bad_kstats = pl.read_database(
        """select * from ranked_view where kstat_rank  <> 1 and is_solution = 1;""",
        connection=con,
    )

    df_bad_fraction_found = pl.read_database(
        """select * from ranked_view where fraction_found_rank  <> 1 and is_solution = 1;""",
        connection=con,
    )


tab2_markdown_text = f"""

## Statistics

* Wordles Solved: {num_solutions}
* Latest Wordle Solved: {max_wordle} on {max_date}
* Failed Wordles (Norm score metric only): {df_bad_norm_scores.shape[0]}
* Failed Wordles (KS Statistic Rank only): {df_bad_kstats.shape[0]}
* Failed Wordles (Fraction Found Rank only): {df_bad_fraction_found.shape[0]}
* Percentage Solved Correctly with Current Algorithm: **{df_pct_correct:.1%}**

"""

result = '✅' if df_last_correct == 1 else '🚫'

tabs = dbc.Tabs(
    [
        dbc.Tab(
            tab1_content,
            label='Explore Solutions',
            tab_id='main-tab',
            id='main-tab',
        ),
        dbc.Tab(
            [
                dcc.Markdown(tab2_markdown_text),
            ],
            label='Statistics',
            tab_id='stat-tab',
        ),
        dbc.Tab(dcc.Markdown(tab_about_content, style={'padding-top': '1em'}), label='About'),
        dbc.Tab(
            html.Div(
                id='bad-solution-tab',
            ),
            label='List of Near-Fails',
        ),
    ],
    active_tab='main-tab',
    id='some-tabs',
)

app.layout = dbc.Container(
    [
        tabs,
        dcc.Location(id='url', refresh=False),
        dbc.Toast(
            dcc.Markdown(
                f'Wordle on {max_date}. Solution was {bool(df_last_correct)} - {result}'
            ),
            id='positioned-toast',
            header='Latest Result',
            is_open=True,
            dismissable=True,
            icon='success' if df_last_correct else 'danger',
            # top: 66 positions the toast below the navbar
            style={'position': 'fixed', 'top': 66, 'right': 50, 'width': 350},
        ),
    ],
)


def wrap_pattern(pattern_tuple, solution, num_pattern_df, impossible_patterns=[]):
    pattern, count = pattern_tuple
    top_words_for_pattern = (
        num_pattern_df.filter(pl.col('pattern').eq(pattern))['guess'].sort().to_list()
    )
    # print(pattern, impossible_patterns)
    return html.Div(
        [
            html.Div(
                html.Span(
                    num_to_color(pattern),
                    id=f'id_{pattern}',
                    style={'font-size': '0.75rem'},
                    className='opacity-100'
                    if (pattern in impossible_patterns or not impossible_patterns)
                    else 'opacity-25',
                ),
            ),
            dbc.Tooltip(
                dcc.Markdown(
                    f"""Count {count}. \n Top Words that make this pattern for **{solution}**: \n"""
                    + '\n'.join([f'* {x}' for x in top_words_for_pattern]),
                    style={'text-align': 'left'},
                ),
                target=f'id_{pattern}',
                placement='auto',
                class_name='text-left',
            ),
        ],
        # className='fs-6',
    )


@callback(
    Output('url', 'search'),
    Input('wordle-input', 'value'),
)
def update_url(wordle_input):
    if not wordle_input:
        raise PreventUpdate
    print(wordle_input)
    word, wordle_id = wordle_input.split('|')
    print(ctx.triggered_id, 'triggered')
    if ctx.triggered_id is not None:
        return '?word=' + word
    else:
        raise PreventUpdate


@callback(
    Output('bad-solution-tab', 'children'),
    Input('some-tabs', 'children'),
)
def make_table_div(_):
    with sqlite3.connect('wordle.db') as con:
        print('reading solutions db')
        df = (
            pl.read_database(
                'select word, norm_score_rank,impossible_pattern_count,cast(metric_sum as INTEGER) metric_sum,metric_sum_rank from ranked_view where is_solution  = 1 and norm_score_rank <> 1 order by 2 DESC',
                connection=con,
            )
            .with_columns(
                word_HREF=(pl.lit('?word=') + pl.col('word')),
            )
            .to_pandas(use_pyarrow_extension_array=True)
        )

    return [
        html.H3('Wordles where the Score-only method failed', style={'padding-top': '1em'}),
        dbc.Table.from_enhanced_dataframe(
            df,
            striped=True,
            style={'width': '75%'},
        ),
    ]


@callback(
    #  Output('pattern-div', 'children', allow_duplicate=True),
    Output('graph-div', 'children'),
    Output('table-div', 'children'),
    Input('url', 'search'),
    State('wordle-input', 'value'),
    prevent_initial_call='initial_duplicate',
)
def update_code(search, word_wordle_tuple):
    if not search:
        raise PreventUpdate
    solution = search.split('=')[-1]
    with sqlite3.connect('wordle.db') as con:
        print(f'querying with {solution}')
        wordle_num = con.execute(
            'select distinct(wordle_id) from solutions where is_solution = 1 and word = ? order by word asc',
            (solution,),
        ).fetchone()[0]
    print('triggered output ')
    if not word_wordle_tuple:
        raise PreventUpdate
    # solution, wordle_num = word_wordle_tuple.split('|')
    wordle_num = int(wordle_num)
    print(wordle_num)
    if not wordle_num:
        return ''

    with sqlite3.connect('wordle.db') as con:
        print('reading solutions db')
        solution_data = pl.read_database(
            'select * from solutions where wordle_id = ? ',
            connection=con,
            execute_options={'parameters': [wordle_num]},
        ).with_columns(
            pl.col('impossible_pattern_list').str.json_decode(
                dtype=pl.List(pl.String),
            )
        )
    print('making figure')

    fig = px.bar(
        solution_data.sort('norm_score')
        .tail(10)
        .with_columns(is_solution=pl.col('is_solution').cast(pl.Boolean)),
        y='word',
        x='norm_score',
        color='is_solution',
        hover_data=['impossible_pattern_count'],
        custom_data=['impossible_pattern_list'],
        #  title='Scored solutions from shared patterns',
        labels={'norm_score': 'Normalized Score', 'word': 'Word'},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    print(f'processing solution data {solution_data.shape}')
    solution_data = (
        solution_data.sort('impossible_pattern_count')
        .with_columns(
            pl.col(['kstatistic']).rank(descending=False, method='min').name.suffix('_rank'),
            pl.col(['fraction_found', 'norm_score'])
            .rank(descending=True)
            .cast(pl.Int16)
            .name.suffix('_rank'),
        )
        .with_columns(
            is_solution=pl.col('is_solution')
            .cast(pl.Boolean)
            .cast(pl.String)
            .str.to_titlecase(),
            metric_sum=pl.col('fraction_found_rank')
            + pl.col('norm_score_rank')
            + pl.col('kstatistic_rank')
            + (pl.col('impossible_pattern_count') + 1) ** 2,
        )
        .sort(
            [
                #     'impossible_pattern_count',
                'metric_sum',
                'norm_score',
            ],
            descending=[False, True],
        )
    )
    cols = [
        'word',
        'total_valid_patterns',
        'valid_patterns_found',
        'impossible_pattern_count',
        'norm_score',
        'fraction_found',
        'kstatistic',
        'is_solution',
    ]
    markdown_text = ''
    check_data = solution_data.sort('norm_score', descending=True)
    if (
        (check_data['norm_score'].shape[0] == 1)
        or (check_data['norm_score'].item(0) - check_data['norm_score'].item(1) < 0.10)
        or (check_data['impossible_pattern_count'].item(0) > 1)
    ):
        cols = [
            'word',
            'impossible_pattern_count',
            'norm_score',
            'fraction_found',
            'kstatistic',
            'fraction_found_rank',
            'norm_score_rank',
            'kstatistic_rank',
            'metric_sum',
            'is_solution',
        ]
        markdown_text = f"""The score-only methodology produced a weak (and possibly wrong) candidate for this dataset. Data was reranked using the "metric sum" which is the sum of:
        
* The rank of my original normalized score metric
* The rank of the [two-sample Kolmogorov-Smirnov test](https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.stats.ks_2samp.html) based on opener frequency
* The rank of the percent of total possible patterns found for the candidate.
* The square of (impossible pattern count + 1)

Ties are broken by the normalized score.

If this seems like a hacky post-hoc heuristic, [it definitely is!](https://marcoshuerta.com/posts/wordle_bluesky)

"""

    print('making table')
    thetable_div = dbc.Table.from_enhanced_dataframe(
        solution_data.select(cols)
        .rename({'word': 'candidate'})
        .to_pandas(use_pyarrow_extension_array=True),
        header_callable=add_header_tooltip,
        striped=True,
    )

    res = [
        [
            make_headline_info_div(
                'Scored solutions from shared patterns',
                'Click to show impossible patterns',
                id='headline-graph',
            ),
            dcc.Graph(figure=fig, id='the-graph'),
        ],
        [dcc.Markdown(markdown_text, style={'text-align': 'left'}), thetable_div],
        #    f'{solution}|{wordle_num}',
    ]
    print('returning from callback')
    return res


@callback(
    Output('dropdown-div', 'children'),
    Output('some-tabs', 'active_tab'),
    Input('main-tab', 'children'),
    Input('url', 'search'),
    State('some-tabs', 'active_tab'),
)
def make_menu(_, search, active_tab):
    print(active_tab)
    if search:
        solution = search.split('=')[-1]
    else:
        solution = 'frank'
    with sqlite3.connect('wordle.db') as con:
        wordle_solutions = pl.read_database(
            'select distinct(word), wordle_id from solutions where is_solution = 1 order by word asc',
            connection=con,
        )
        options_wordle = [
            {'label': label, 'value': f'{label}|{value}'}
            for label, value in wordle_solutions.select('word', 'wordle_id').iter_rows()
        ]
        wordle_num = con.execute(
            'select distinct(wordle_id) from solutions where is_solution = 1 and word = ? order by word asc',
            (solution,),
        ).fetchone()[0]
    if search:
        value = f'{solution}|{wordle_num}'
    else:
        value = 'frank|821'
    return (
        dbc.InputGroup(
            [
                dbc.InputGroupText('Chooose a Wordle Solution: '),
                dbc.Select(
                    id='wordle-input',
                    placeholder='Choose or Enter Wordle Solution',
                    options=options_wordle,
                    value=value,
                    #  style={'width': '75%'},
                ),
            ],
            style={
                'width': '55%',
                'margin': 'auto',
                'padding-bottom': '2em',
                'padding-top': '1em',
            },
        ),
        'main-tab',
    )


@callback(
    Output('pattern-div', 'children', allow_duplicate=False),
    Output('weird-guesses', 'children'),
    Input('the-graph', 'clickData'),
    State('wordle-input', 'value'),
    prevent_initial_call=False,
)
def update_pattern_on_click(click_data, word_wordle_tuple):
    print('triggeing')
    solution, wordle_num = word_wordle_tuple.split('|')
    wordle_num = int(wordle_num)
    patterns = []
    label = solution
    if click_data:
        patterns = click_data['points'][0]['customdata'][0]
        label = click_data['points'][0]['label']
    with sqlite3.connect('wordle.db') as con:
        num_pattern_df = pl.read_database(
            'select pc.pattern,count,freq,wordle_id,guess from pattern_counts pc left join patterns p on pc.pattern = p.pattern and target = ? where pc.wordle_id = ? order by 1 desc;',
            connection=con,
            execute_options={'parameters': [solution, wordle_num]},
        )
        post_count = con.execute(
            'select count(*) from posts where wordle_id = ? ', (wordle_num,)
        ).fetchone()[0]
    print(num_pattern_df.shape)
    num_pattern_list = sorted(
        [
            (key, val)
            for key, val in num_pattern_df.unique(subset=['pattern', 'count'])
            .select(['pattern', 'count'])
            .iter_rows()
        ],
        key=lambda x: (-x[1], x[0]),
    )
    if solution == 'grace':
        print(num_pattern_list, len(num_pattern_list))

    weird_word_list = (
        num_pattern_df.with_columns(max_freq=pl.col('freq').max().over('pattern'))
        .filter(pl.col('freq').lt(1e5) & pl.col('count').eq(1) & pl.col('max_freq').lt(1e5))[
            'guess'
        ]
        .to_list()
    )
    augmented_list = [f'*{x}*' for x in weird_word_list]
    weird_guess_children = dcc.Markdown(
        '**Unusual Implied Guesses:** '
        + (', '.join(augmented_list) if augmented_list else '*None.*')
    )

    # weird_guesses = num_pattern_df.filter(pl.col('count').eq(1) & pl.col('freq').eq(0))['word'].to_list()
    res = (
        [
            make_headline_info_div(
                f'Highlighting Impossible Patterns for {label.upper()}',
                f'These are "impossible" patterns that can not be made if **{label}** was the solution (indicating that it is likely not the solution)',
                id='highlight-impossible',
            )
            if patterns
            else make_headline_info_div(
                'Showing all Shared Patterns',
                'Patterns found on Blue Sky shared for this Wordle. Hover to see the number of times the pattern appeared, and common words associated with this pattern for the solution.',
                id='standard-all-shared-patterns',
            ),
            html.Div(
                [
                    wrap_pattern(
                        x,
                        solution=solution,
                        impossible_patterns=patterns,
                        num_pattern_df=num_pattern_df,
                    )
                    for x in num_pattern_list
                ],
                style={
                    'display': 'flex',
                    'flex-wrap': 'wrap',
                    'gap': '5px',
                    'margin': '5px',
                },
            ),
            dcc.Markdown(f'Solved with {post_count} total posts.'),
        ],
        weird_guess_children,
    )

    return res


server = app.server

if __name__ == '__main__':
    app.run(debug=True)
