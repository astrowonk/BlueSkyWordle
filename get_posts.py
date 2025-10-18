from atproto import Client


import polars as pl
import sqlite3

from config import bluesky_login, bluesky_password
from time import sleep


def get_bluesky_posts(wordle_num, max_posts=400, refresh_data=False):
    """Helper function, requires searchtweets v2 and proper Twitter api credentials. Turns a single wordle
    query into a dataframe that can be used inside the TwitterWordle class."""
    with sqlite3.connect('wordle.db') as con:
        df = pl.read_database(
            'select * from posts where wordle_id = ?',
            execute_options={'parameters': [wordle_num]},
            connection=con,
        )
    if not df.is_empty() and not refresh_data:
        print('retrieved from database')
        return df
    client = Client()
    _ = client.login(bluesky_login, bluesky_password)
    wordle_num_str = f'{wordle_num:,}'
    out = client.app.bsky.feed.search_posts({'q': f'wordle {wordle_num_str}', 'limit': 100})
    orig = [(x['record']['text'], wordle_num_str) for x in out.model_dump()['posts']]
    for cursor in [str(x) for x in (range(99, max_posts - 100, 100))]:
        out = client.app.bsky.feed.search_posts(
            {'q': f'wordle {wordle_num_str}', 'limit': 100, 'cursor': cursor}
        )
        new = [(x['record']['text'], wordle_num_str) for x in out.model_dump()['posts']]
        orig.extend(new)
    orig = list(set(orig))
    df = pl.DataFrame([{'post_text': x, 'wordle_id': wordle_num} for x, y in orig])
    with sqlite3.connect('wordle.db') as con:
        con.execute('delete from posts where wordle_id = ?', (wordle_num,))
    df.write_database('posts', connection='sqlite:///wordle.db', if_table_exists='append')
    # sleep(5)
    return df
