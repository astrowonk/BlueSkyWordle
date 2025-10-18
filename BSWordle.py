"""Hlep this is a class"""

import datetime
import hashlib
import json
import sqlite3
import gzip
from collections import Counter, defaultdict

import pandas as pd
import plotly.express as px
import polars as pl
import requests
from get_posts import get_bluesky_posts
from scipy.stats import ks_2samp, mannwhitneyu
from tqdm.autonotebook import tqdm


def num_to_color(pattern: str):
    return pattern.replace('0', '⬛').replace('1', '🟨').replace('2', '🟩')


def get_date(wordle_num):
    return (
        datetime.datetime(2021, 6, 19) + datetime.timedelta(days=int(wordle_num))
    ).strftime('%Y-%m-%d')


def help_hash(word):
    return hashlib.sha256(word.encode()).hexdigest()[:10]


class BlueskyWordle:
    """my wordle solving class"""

    best_df = None
    guess_list = None
    pattern_counter = None
    snapshot_df_list = None
    snapshot_df = None
    best_penalty = None
    unhashed_prediction = None
    filter_posts = True

    def __init__(self, use_limited_targets: bool = True, filter_posts=True):
        """_summary_

        Parameters
        ----------
        use_limited_targets : bool, optional
            use smaller solution dictionary, by default True
        """
        self.filter_posts = filter_posts
        file_name = 'all-words-counters-2025.json.gz'
        if use_limited_targets:
            file_name = 'zipped_counters_nyt_2025_update.json'
        if file_name.endswith('.gz'):
            with gzip.open(file_name, 'r') as f:
                self.zipped_counters = json.load(f)
        else:
            with open(file_name, 'r') as f:
                self.zipped_counters = json.load(f)
        print(
            f'Loaded {len(self.zipped_counters)} pre-computed lookup dictionaries from {file_name}.'
        )
        with open('counters-openers-data-all.json', 'r') as f:
            self.counter_data_openers = json.load(f)
        self.hashed_keys = [help_hash(x) for x in self.zipped_counters.keys()]
        self.df = None

    def dataframe_to_list(self, df):
        self.df = (
            df.with_columns(
                post_list=pl.col('post_text')
                .str.replace_all('🟩', '2')
                .str.replace_all('🟨', '1')
                .str.replace_all(r'[⬛⬜]', 0)
                .str.extract_all('([012]{5})')
            )
            .with_row_index()
            .filter(pl.col('post_list').list.len().le(6))
            .filter(pl.col('post_text').str.count_matches('[🟩🟨⬛⬜]').le(30))
            .filter(pl.col('post_text').str.count_matches(r'🟩🟩🟩🟩🟩').le(1))
            .with_columns(first_guess=pl.col('post_list').list.first())
            #  .filter(pl.col('post_text').str.count_matches('\n').le(10))
        )
        if self.filter_posts:
            self.df = (
                self.df.remove(pl.col('post_text').str.contains('wordle.at'))
                .remove(pl.col('post_text').str.contains('gridgames'))
                .remove(pl.col('post_text').str.contains('welt.de'))
                .remove(pl.col('post_text').str.contains('🔥'))
                .remove(
                    pl.col('post_text')
                    .str.to_lowercase()
                    .str.count_matches(r'(?i)\bwordle\s+\d{1,3}(?:,\d{3}?|\d+)\b')
                    .gt(1)
                )
                .remove(pl.col('post_text').str.to_lowercase().str.contains('wordled'))
            )
        self.first_guesses = self.df['first_guess'].explode().drop_nulls().to_list()
        print(f'Bulding pattern list for {self.wordle_num} using {self.df.shape[0]} posts.')

        return (
            self.df['post_list']
            .explode()
            .drop_nulls()
            # .unique()
            .to_list()
        )

    @staticmethod
    def make_df_from_dict(res_dict):
        return pl.DataFrame(
            [
                {
                    'word': word,
                    'score': d['score'],
                    'total_valid_patterns': d['total_valid_patterns'],
                    #'total_possible_score': d['ftotal_possible_score'],
                    'valid_patterns_found': d['valid_patterns_found'],
                    'impossible_pattern_count': d['impossible_pattern_count'],
                    'impossible_pattern_list': d['bad_pattern_list'],
                    #    'strongest_signal_patterns': [x[1] for x in d['strongest_signals']],
                }
                for word, d in res_dict.items()
            ]
        )

    def build_counter(self, df):
        guess_list = self.dataframe_to_list(df)

        guess_list = [x for x in guess_list if x not in ['22222']]
        self.guess_list = guess_list

        self.pattern_counter = Counter(guess_list)

    def build_score_frame(
        self, df, min_count=1, max_count=10000, build_snapshot: bool = False
    ):
        assert len(df['wordle_id'].unique()) == 1, (
            'Solve one works on a dataframe containing 1 wordle only.'
        )

        if build_snapshot:
            assert self.best_penalty, (
                'Wordle must have been solved first with build_snapshot = True'
            )
        self.build_counter(df)

        res_dict = {}
        self.snapshot_df_list = []
        for word, d in self.zipped_counters.items():
            res_dict[word] = defaultdict(int)
            res_dict[word]['bad_pattern_list'] = []
            res_dict[word]['total_valid_patterns'] = len(d)
        # res_dict[word]['total_possible_score'] = sum(d.values())

        for i, (pattern, guess_count) in enumerate(self.pattern_counter.most_common()):
            #   res_dict[word]['strongest_signals'] = []
            for word, d in self.zipped_counters.items():
                if max_count >= guess_count >= min_count:
                    res = d.get(pattern)
                    if res is not None:
                        res_dict[word]['score'] += res
                        res_dict[word]['valid_patterns_found'] += 1
                    else:
                        res_dict[word]['impossible_pattern_count'] += max(1, guess_count - 1)
                        res_dict[word]['bad_pattern_list'].append(pattern)
                        if build_snapshot and self.best_penalty:
                            res_dict[word]['score'] += self.best_penalty
            if build_snapshot:
                self.snapshot_df_list.append(
                    self.make_df_from_dict(res_dict)
                    .sort('score', descending=True)
                    .head(10)
                    .with_columns(
                        iteration=i,
                        pattern=pl.lit(num_to_color(pattern)),
                        is_solution=pl.col('word')
                        .map_elements(help_hash, return_dtype=pl.String)
                        .eq(self.hashed_solution)
                        .cast(pl.Int8),
                    )
                )

        return self.make_df_from_dict(res_dict)

    def show_impossible(self, word=None):
        if not word:
            thelist = self.best_df['impossible_pattern_list'].to_list()[0]
        else:
            thelist = self.best_df.filter(pl.col('word').eq(word))[
                'impossible_pattern_list'
            ].to_list()[0]

        for x in self.df.filter(
            pl.col('post_list').list.set_intersection(thelist).list.len().gt(0)
        )['post_text']:
            print(x)
        return thelist

    def plot(self):
        return px.bar(
            self.best_df.sort('norm_score').tail(10),
            y='word',
            x='norm_score',
            hover_data=['impossible_pattern_list'],
            labels={'norm_score': 'Normalized Score', 'word': 'Word'},
        )

    def get_solution_from_api(self, wordle_num):
        date = get_date(wordle_num)
        url = f'https://www.nytimes.com/svc/wordle/v2/{date}.json'
        print(f'Looking up solution from API for {date}')
        data = requests.get(url).json()
        wordle_num = data['days_since_launch']
        target_word = data['solution']
        return target_word

    def get_solution_from_db(self, wordle_num):
        with sqlite3.connect('wordle.db') as con:
            res = con.execute(
                'select solution from hashed_solutions where wordle_id = ? ', (wordle_num,)
            ).fetchone()
        if res is not None:
            return res[0]

    def load_solution(self, wordle_num):
        self.hashed_solution = None
        solution = self.get_solution_from_db(wordle_num)
        if solution is not None:
            self.hashed_solution = solution
        else:
            self.hashed_solution = help_hash(self.get_solution_from_api(wordle_num))
            with sqlite3.connect('wordle.db') as con:
                print(f'Saving hashed wordle solution for {wordle_num}')
                con.execute(
                    'insert into hashed_solutions VALUES (?,?) ',
                    (self.hashed_solution, wordle_num),
                )
        if self.hashed_solution not in self.hashed_keys:
            print('** Impossible to solve with loaded dictionary. Solution data missing. **')
            return False
        return True

    def verify_solution(self, prediction):
        return self.hashed_solution == help_hash(prediction)

    def build_data_only(self, wordle_num):
        self.wordle_num = wordle_num

        df = get_bluesky_posts(wordle_num, max_posts=900)
        self.build_counter(df)

    def solve(
        self,
        wordle_num,
        min_count=1,
        downsample=None,
        filter_pattern=True,
        mask_result=False,
        plot=False,
        build_snapshot=False,
        max_count=10000,
        resort_poor_score=True,
        force_resort=False,
    ) -> bool:
        """Solve wordle from a wordle number"""
        self.wordle_num = wordle_num
        res = self.load_solution(wordle_num)
        if not res and len(self.zipped_counters) < 5000:
            print(f'Loading larger dictionary for {wordle_num} ')
            self.__init__(use_limited_targets=False)
            res = self.load_solution(wordle_num)
            if not res:
                print('Cannot be solved with larger dictionary.')
                return False
        df = get_bluesky_posts(wordle_num, max_posts=800)
        if downsample:
            df = df.sample(downsample)

        return self.solve_df(
            df,
            min_count=min_count,
            filter_pattern=filter_pattern,
            mask_result=mask_result,
            plot=plot,
            build_snapshot=build_snapshot,
            max_count=max_count,
            resort_poor_score=resort_poor_score,
            force_resort=force_resort,
        )

    def solve_df(
        self,
        df,
        min_count=1,
        filter_pattern=True,
        mask_result=False,
        plot=False,
        build_snapshot=False,
        max_count=10000,
        resort_poor_score=True,
        force_resort=False,
    ) -> bool:
        """Solve a dataframe"""
        wordle_num = int(df['wordle_id'].unique()[0])
        if filter_pattern:
            df = df.filter(
                pl.col('post_text').str.to_lowercase().str.contains(f'wordle {wordle_num:,}')
            )
        solution_df = self.build_score_frame(
            df,
            min_count=min_count,
            build_snapshot=build_snapshot,
            max_count=max_count,
        )
        # self.solution_df = solution_df

        best_delta = 0
        if build_snapshot:
            print('Snapshot built.')
            self.snapshot_df = pl.concat(self.snapshot_df_list, how='diagonal_relaxed')

            return
        for p in range(-7, -300, -5):
            penalty = p * 1e7
            res = (
                solution_df.with_columns(
                    adjusted_score=pl.col('score')
                    + pl.col('impossible_pattern_count') * penalty
                )
                .with_columns(
                    norm_score=(pl.col('adjusted_score') - pl.col('adjusted_score').mean())
                    / pl.col('adjusted_score').std()
                )
                .sort('norm_score', descending=True)
            ).head(50)  # changed to 50 because of wordle 821!
            delta = res['norm_score'].item(0) - res['norm_score'].item(1)
            # print(delta)
            if delta > best_delta:
                #  print(best_delta)
                best_df = res
                best_delta = delta
                self.best_penalty = penalty

        best_df = best_df.with_columns(
            fraction_found=pl.col('valid_patterns_found') / pl.col('total_valid_patterns')
        ).with_columns(
            kstatistic=pl.col('word').map_elements(self.ks_stat, return_dtype=pl.Float64),
            #  mwstatistic=pl.col('word').map_elements(self.mw_stat, return_dtype=pl.Float64),
        )

        if force_resort or (
            (best_delta < 0.10 or best_df['impossible_pattern_count'].item(0) > 0)
            and resort_poor_score
            #    and best_df['impossible_pattern_count'].mean() > 5
            #    and best_df['impossible_pattern_count'].item(0) > 0
        ):
            print('Low score metric. Resorting.')
            best_df = (
                best_df.filter(pl.col('impossible_pattern_count').le(50))
                .sort('impossible_pattern_count')
                .head(10)
                .with_columns(
                    pl.col(['kstatistic'])
                    .rank(descending=False, method='min')
                    .name.suffix('_rank'),
                    pl.col(['fraction_found', 'norm_score'])
                    .rank(descending=True)
                    .name.suffix('_rank'),
                )
                .with_columns(
                    metric_sum=pl.col('fraction_found_rank')
                    + pl.col('norm_score_rank')
                    + pl.col('kstatistic_rank')
                    + (pl.col('impossible_pattern_count') + 1) ** 2
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
        prediction = best_df['word'].item(0)
        best_impossible_pattern = best_df['impossible_pattern_count'].item(0)
        if best_impossible_pattern > 1 and min_count == 1:
            # look at this hack for wordle 800. some weirdo fake patterns in the dataset.
            print(
                f'Impossible pattern for {prediction} = {best_impossible_pattern}. Rerunning with higher minimum.'
            )
            return self.solve_df(
                df, min_count=2, resort_poor_score=resort_poor_score, mask_result=mask_result
            )
        result = self.verify_solution(prediction)
        hash_print_text = ''
        if mask_result:
            prediction = help_hash(prediction)
            hash_print_text = 'SHA Hash of '
        print(
            f'{hash_print_text}Prediction: {prediction.upper()}.\nDelta over runner-up: {best_delta:.2f}'
        )
        print(
            f'{best_df["valid_patterns_found"].item(0)} of {best_df["total_valid_patterns"].item(0)} possible patterns found ({best_df["fraction_found"].item(0):.2%}).'
        )
        print(f'Impossibe pattern count: {best_impossible_pattern}')
        self.best_df = best_df.with_columns(
            hashed_word=pl.col('word').map_elements(help_hash, return_dtype=pl.String)
        ).with_columns(
            is_solution=pl.col('hashed_word').eq(self.hashed_solution).cast(pl.Int8)
        )
        if mask_result:
            self.best_df = self.best_df.with_columns(word=pl.col('hashed_word'))
        if plot:
            return self.make_plot(self.best_df.sort('norm_score'))
        print(f'Result confirmation: {result}')
        return result

    def snapshot_plot(self, min_iteration=50):
        """make animated snapshot plot"""
        assert self.snapshot_df_list
        thedf = self.snapshot_df.filter(pl.col('iteration').ge(min_iteration)).sort(
            ['iteration', 'score'], descending=False
        )
        return px.bar(
            thedf,
            y='word',
            x='score',
            hover_data=['impossible_pattern_list', 'valid_patterns_found'],
            range_x=(min(thedf['score']), max(thedf['score'])),
            animation_frame='pattern',
            color='is_solution',
            range_color=[0, 1],
            # color_discrete_map={'true': 'green', 'false': 'red'},
        )

    def word_to_distribution(self, word):
        out = []
        thedict = (
            (
                pd.Series(
                    {
                        key: val
                        for key, val in self.counter_data_openers.get(word, {}).items()
                        if key != '22222'
                    }
                )
                // 100
                - 1
            )
            .astype(int)
            .to_dict()
        )
        for key, val in thedict.items():
            if key != '22222':
                out.extend([key] * val)
        return out

    def ks_stat(self, word):
        return ks_2samp(self.first_guesses, self.word_to_distribution(word)).statistic
