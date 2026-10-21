"""Helper functions"""

import heapq
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import polars as pl
from tqdm.autonotebook import tqdm


def get_num_line(guess, answer):
    """Make the wordle score line for a given guess and answer, method borrowed from my Wordle solver class"""
    match_and_position = [
        2 * int(letter == answer[i]) for i, letter in enumerate(guess)
    ]
    remaining_letters = [x for i, x in enumerate(answer) if match_and_position[i] != 2]

    # print('remaining letters', remaining_letters)

    def find_non_position_match(remaining_letters, guess):
        """has to be a better way"""
        res = []
        for i, letter in enumerate(guess):
            # print(letter)
            # print(letter in remaining_letters)
            if letter in remaining_letters and match_and_position[i] != 2:
                res.append(1)
                remaining_letters.remove(letter)
            else:
                res.append(0)
        return res

    non_position_match = find_non_position_match(remaining_letters, guess)
    return [x or y for x, y in zip(match_and_position, non_position_match)]


def stringify(x):
    return ''.join(str(y) for y in x)


def make_freqs(
    filename='wordle-dictionary-full.txt', frequency_data='unigram_freq.csv'
):
    short_words = pd.read_csv(filename, header=None)[0].tolist()
    df = pd.read_csv(frequency_data)
    # Establish a minimum frequency for any Wordle word that's missing from the frequency dataset
    min_freq = 0
    english_freqs = {df['word'][i]: df['count'][i] for i in df.index}
    return {w: int(english_freqs.get(w, min_freq)) for w in short_words}


def process_result2(r, freqs=None):
    temp_list = [{'str_score': stringify(x['score']), 'guess': x['guess']} for x in r]
    df = pd.DataFrame(temp_list)
    df['guess_rank'] = df['guess'].apply(lambda x: freqs.get(x, 0))
    out = df.groupby('str_score')['guess_rank'].sum()
    return out.to_dict()


def helper_func(target_word, freqs, short_words):
    return process_result2(
        [
            {'score': get_num_line(x, target_word), 'target': target_word, 'guess': x}
            for x in short_words
        ],
        freqs,
    )


_worker_words = None
_worker_freqs = None


def _init_top_words_worker(all_words, new_freqs):
    global _worker_words, _worker_freqs
    _worker_words = all_words
    _worker_freqs = new_freqs


def _top_words_for_target(target_word):
    pattern_freqs = defaultdict(list)
    for word in _worker_words:
        freq = _worker_freqs.get(word, 0)
        pattern = stringify(get_num_line(word, target_word))
        heap = pattern_freqs[pattern]
        heapq.heappush(heap, (-freq, word))
        pattern_freqs[pattern] = heapq.nsmallest(10, heap)
    return target_word, pattern_freqs


def make_top_words_for_pattern(target_words, all_words, new_freqs):
    # Nested dict: {target_word: {pattern: [(freq, word)]}}
    top_freqs = defaultdict(lambda: defaultdict(list))

    with ProcessPoolExecutor(
        initializer=_init_top_words_worker,
        initargs=(all_words, new_freqs),
        max_workers=8,
    ) as executor:
        for target_word, pattern_dict in tqdm(
            executor.map(_top_words_for_target, target_words),
            total=len(target_words),
        ):
            top_freqs[target_word].update(pattern_dict)

    out = []
    for target_word, pattern_dict in tqdm(top_freqs.items()):
        for pattern, thelist in pattern_dict.items():
            for freq, guess in thelist:
                out.append(
                    {
                        'pattern': pattern,
                        'target': target_word,
                        'guess': guess,
                        'freq': -freq,
                    }
                )
    return pl.DataFrame(out)
