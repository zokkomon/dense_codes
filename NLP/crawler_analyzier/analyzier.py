#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 22:11:44 2023

@author: zok
"""

import os
import glob
import warnings
import nltk
import textblob
import pandas as pd
import numpy as np

from nltk.tokenize import sent_tokenize
from nltk.tokenize import SyllableTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import subjectivity
from nltk.corpus import cmudict

from text_extractor import Scraper
warnings.filterwarnings('ignore')

def get_stopwords(directory_path):
    files = glob.glob(os.path.join(directory_path, '*.txt'))
    
    stopwords_set = set()
    for file in files:          
        with open(file, 'rb') as f:
            stopwords_set.update(f.read().decode('latin-1').split())
    
    # Convert the set of stopwords to a list
#     stopwords_list = list(stopwords_set)

    return stopwords_set

def get_files(directory_path, py=True):
    files = glob.glob(os.path.join(directory_path, '*.txt'))

    if py:
        with open(files[0], 'r') as f:
            positive_words = f.read().split()
        return positive_words
        
    else:
        with open(files[1], 'r') as f:
            negative_words = f.read().split()
        return negative_words
    
def calculate_sentiment_scores(text, positive_words, negative_words):
    positive_score = 0
    negative_score = 0

    for word in text.split():
        if word in positive_words:
            positive_score += 1
        
        elif word in negative_words:
            negative_score -= 1  # Multiply by -1 to make it a positive number

    return positive_score, negative_score


# Function to analyze text
def analyze_text(directory, custom_stopwords):
    
    results= []
    # Loop through each file in the directory
    filename, url = Scraper().text_extractor()
    # Loop through each file in the directory
    for idx,filena in enumerate(os.listdir(directory)):
        
        if filename[idx].endswith(".txt"):
            file_path = os.path.join(directory, filename[idx])
            with open(file_path, "r") as file:
              
                text = file.read()
                words = text.split()
                if len(words) == 0 in filename:
                    print("No words found in file")
                    return None
                    continue
                # sentences = sent_tokenize(text)
                
                # Calculate sentiment scores
                Positive_Score, Negative_Score = calculate_sentiment_scores(text, pos_words, neg_words)
                polarity_score = (np.exp(2 * (Positive_Score - Negative_Score)/ ((Positive_Score + Negative_Score) + 0.000001)) - 1) / (np.exp(2 * (Positive_Score - Negative_Score)/ ((Positive_Score + Negative_Score) + 0.000001)) + 1)
                subjectivity_score = 1 / (1 + np.exp(-(Positive_Score + Negative_Score)/ ((len(text)) + 0.000001)))
#                 sia = SentimentIntensityAnalyzer()
#                 sentiment_scores = sia.polarity_scores(text)

#                 # Get positive, negative, and polarity scores
#                 Positive_Score = sentiment_scores["pos"]
#                 Negative_Score = sentiment_scores["neg"]
#                 polarity_score = sentiment_scores["compound"]
#                 subjectivity_score = len([word for word in words if word in subjectivity.words()]) / len(words)

                # Calculate readability metrics
                avg_sentence_length = len(text.split()) / len(text.splitlines())
                complex_words = [word for word in text.split() if len(word) >= 3]
                percentage_complex_words = len(complex_words) / len(text.split()) * 100
                fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
                #Average Number of Words Per Sentence
                avg_words_per_sentence = len(text.split()) / len(text.splitlines())
                #Complex Word Count
                complex_word_count = len(complex_words)
                word_count = len(text.split())

                # Calculate syllable count
                syllable_count = sum([len(SyllableTokenizer().tokenize(word)) for word in text.split()])
                syllables_per_word = syllable_count / word_count

                # Calculate personal pronouns
                pronouns = ['i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 'hers', 'it', 'its', 'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs']
                personal_pronoun_count = len([word for word in text.split() if word.lower() in pronouns and word.lower() not in custom_stopwords])

                # Calculate average word length
                avg_word_length = sum([len(word) for word in text.split()]) / len(text.split())

                # Store results in a dictionary
                result = {
                    "URL_ID": filename[idx],
                    "URL" : url[idx],
                    "POSITIVE SCORE" : Positive_Score,
                    "NEGATIVE SCORE" : Negative_Score,
                    "POLARITY SCORE": polarity_score,
                    "SUBJECTIVITY SCORE": subjectivity_score,
                    "AVG SENTENCE LENGTH": avg_sentence_length,
                    "PERCENTAGE OF COMPLEX WORDS": percentage_complex_words,
                    "FOG INDEX": fog_index,
                    "AVG NUMBER OF WORDS PER SENTENCE": avg_words_per_sentence,
                    "COMPLEX WORD COUNT": complex_word_count,
                    "WORD COUNT": word_count,
                    "SYLLABLE PER WORD": syllables_per_word,
                    "PERSONAL PRONOUNS": personal_pronoun_count,
                    "AVG WORD LENGTH": avg_word_length
                    }
                results.append(result)
                
    # # Sort the list of results by filename
    # sorted_results = sorted(results, key=lambda x: x['URL_ID'])

    analyze_text_results = pd.DataFrame(results)
    return analyze_text_results

if __name__ == "__main__":
    pos_words = get_files("/home/zok/joker/blackcoffer/MasterDictionary/")
    neg_words = get_files("/home/zok/joker/blackcoffer/MasterDictionary/", py=False)
    
    custom_stopwords = get_stopwords("/home/zok/joker/blackcoffer/StopWords/")
    
    analyze_text_results = analyze_text("/home/zok/joker/blackcoffer/text_files/", custom_stopwords)
    analyze_text_results.to_excel('analyzed.xlsx')