from election_text_analysis import election_text_analysis, read_data, analyze
import pandas as pd

import pytest

def test_read_timeseries():    
    '''
        Tests reading in the timeseries data
    '''
    df = read_data.read_timeseries_data()

    # Checks that we have a valid DataFrame
    assert type(df) == pd.DataFrame

    # We should have 68224 rows by 1030 columns
    assert df.shape == (68224, 1030)    

def test_read_open_ends():    
    '''
        Tests reading in the open-ended data
    '''
    open_ends_2020 = read_data.read_2020_open_ends()   
    # We should have 8280 rows by 8 columns
    assert open_ends_2020.shape == (8280, 8) 

    open_ends_2016 = read_data.read_2016_open_ends()   
    # We should have 4271 rows by 8 columns
    assert open_ends_2016.shape == (4271, 8)

    open_ends_2012 = read_data.read_2012_open_ends()   
    # We should have 5914 rows by 8 columns
    assert open_ends_2012.shape == (5914, 8)

    open_ends_2008 = read_data.read_2008_open_ends()   
    # We should have 2323 rows by 8 columns
    assert open_ends_2008.shape == (2323, 8)

def test_read_overall():
    '''
        Tests reading in the overall data file
    '''
    df = read_data.read_all_data()
    assert df.shape == (68224, 1038)

def test_get_unique_words():
    # Tests getting a list of words, without any stopwords
    assert sorted(analyze.get_unique_words("This is a sample sentence", stopwords=[])) == sorted(['sentence', 'sample', 'this'])

    # Tests getting a list of words, with stopwords
    assert sorted(analyze.get_unique_words("This is a sample sentence")) == sorted(['sentence', 'sample'])

    # Tests getting a list of unique words from a sentence with duplicates
    assert sorted(analyze.get_unique_words("This sentence has duplicated duplicated words")) == sorted(['sentence', 'words', 'duplicated'])

def test_get_word_frequencies():
    sentences = pd.Series(["This is a sentence, it is a long sentence...", "This is another sentence", "A third sentence", "A fourth sentence"])
    assert analyze.get_word_frequencies(sentences) == {'long': 1.0/4, 'sentence': 1.0, 'another': 1.0/4, 'third': 1.0/4, 'fourth': 1.0/4}

def test_compare_word_frequencies():
    group_1_series = pd.Series(["This is a sentence, it is a long sentence...", "This is another sentence", "A third sentence", "A fourth sentence"])
    group_2_series = pd.Series(["These are words", "This is also a sentence", "All of these are words happily", "Yet more words"])
    results = analyze.compare_word_frequencies(group_1_series, group_2_series)
    assert results[0] == {'word': 'words', 'group_1': 0.0, 'group_2': 0.75, 'delta': -0.75}
    assert results[-1] ==  {'word': 'sentence', 'group_1': 1.0, 'group_2': 0.25, 'delta': 0.75}

    


