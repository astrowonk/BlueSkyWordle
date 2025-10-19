"""Helper functions"""

import pandas as pd
import polars as pl
import heapq
from collections import defaultdict
from tqdm.autonotebook import tqdm


def get_num_line(guess, answer):
    """Make the wordle score line for a given guess and answer, method borrowed from my Wordle solver class"""
    match_and_position = [2 * int(letter == answer[i]) for i, letter in enumerate(guess)]
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


def make_freqs(filename='wordle-dictionary-full.txt', frequency_data='unigram_freq.csv'):
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


def make_top_words_for_pattern(target_words, all_words, new_freqs):
    # Nested dict: {target_word: {pattern: [(freq, word)]}}
    top_freqs = defaultdict(lambda: defaultdict(list))

    for target_word in tqdm(target_words):
        for word in all_words:
            freq = new_freqs.get(word, 0)
            pattern = stringify(get_num_line(word, target_word))
            heap = top_freqs[target_word][pattern]
            _ = heapq.heappush(heap, (-freq, word))
            top_freqs[target_word][pattern] = heapq.nsmallest(10, heap)

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
