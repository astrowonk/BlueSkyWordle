import argparse
import datetime
import json
import sqlite3

import polars as pl
from BSWordle import BlueskyWordle
from tqdm import tqdm
from dash_app import get_date


def myjson(x):
    return json.dumps(list(x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_num', type=int, nargs='?', default=None)
    parser.add_argument('stop_num', type=int, nargs='?', default=None)
    parser.add_argument('--skip-existing', action='store_true', default=False)
    args = parser.parse_args()

    with sqlite3.connect('wordle.db') as con:
        df = pl.read_database(
            'select max(wordle_id) wordle_id,count(*) as count from solutions where is_solution = 1 order by 1 asc',
            connection=con,
        )
        max_wordle = df['wordle_id'].item(0)
        num_solutions = df['count'].item(0)
        max_date = get_date(wordle_num=max_wordle)
    todays_wordle_num = int(
        (datetime.datetime.today() - datetime.datetime(2021, 6, 19)).total_seconds() // 86400
    )
    if args.start_num:
        start_num = args.start_num
    else:
        start_num = max_wordle
        stop_num = todays_wordle_num
    if args.stop_num:
        assert args.start_num
        stop_num = args.start_num

    failed_list = []
    for wordle_num in tqdm(range(start_num, stop_num + 1)):
        bsw = BlueskyWordle(use_limited_targets=True)
        if args.skip_existing:
            with sqlite3.connect('wordle.db') as con:
                res = con.execute(
                    'select * from solutions where wordle_id = ?;', (wordle_num,)
                ).fetchall()
            if res:
                print(f'Skipping {wordle_num}')
                continue

        with sqlite3.connect('wordle.db') as con:
            try:
                con.execute('delete from solutions where wordle_id = ?;', (wordle_num,))
            except sqlite3.Error:
                print('delete failed')
        res = bsw.solve(wordle_num)
        if not res:
            failed_list.append(wordle_num)
        bsw.best_df.with_columns(
            wordle_id=wordle_num,
            impossible_pattern_list=pl.col('impossible_pattern_list').map_elements(
                myjson, return_dtype=pl.String
            ),
        ).head(10).drop(
            [
                'kstatistic_rank',
                'fraction_found_rank',
                'norm_score_rank',
                'metric_sum',
            ],
            strict=False,
        ).write_database(
            'solutions', connection='sqlite:///wordle.db', if_table_exists='append'
        )
        with sqlite3.connect('wordle.db') as con:
            try:
                con.execute('delete from pattern_counts where wordle_id = ?;', (wordle_num,))
            except sqlite3.Error:
                print('delete failed')
        num_pattern_df = pl.DataFrame([
            {'pattern': key, 'count': val, 'wordle_id': wordle_num}
            for key, val in bsw.pattern_counter.items()
        ])
        num_pattern_df.write_database(
            'pattern_counts', connection='sqlite:///wordle.db', if_table_exists='append'
        )

    if failed_list:
        print(f'Failed to solve {",".join([str(x) for x in failed_list])}')
    else:
        print('All solved correctly')
