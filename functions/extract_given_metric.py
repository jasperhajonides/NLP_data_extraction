#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 14:53:31 2021

@author: jasperhajonides
"""

import sys
import os
import pickle
import pandas as pd
sys.path.append('../scripts/')
from preprocessing import page_highest_occurance, predict_best_page
from preprocessing import create_match_dataframe, define_company_dictionary
from spacy_model import spacy_probabilities, spacy_predict

def extract_given_metric(path, metric):
    """
    This function returns a dataframe indicating all classified occurences
    of a given metric with page locations and matched text
    within a file located at the given path.

    ---------------
    Parameters:
        metric: "n_employees" "gender_diversity", "ethnic_diversity"...

    *** NOTE: metric must already be implemented in preprocessing.py ***
    RETURN
        dataframe

    Jasper Hajonides (31082021)
    """
    # run spaCy token/phrase matcher on document and return phrases.
    dfr = create_match_dataframe(path, metric)
    company_info = define_company_dictionary(path)

    #If no matches found, classify metric as not found
    if len(dfr)==0:
        print("No potential instances found for {}".format(metric), end ='\n\n')
        df_final=pd.DataFrame({"filename":[company_info['filename']],"name":[company_info['name']],
                         "year":[company_info['year']], "class":[0],"predicted_page":[[]],
                         "predicted_text":[[]]})
    else:

        print("Accessing saved {0} model...".format(metric))
        # load in classifier (sklearn/spaCy)
        pkl_filename = r'../models/{0}_model.pkl'.format(metric)
        with open(pkl_filename, "rb") as file:
            classifier = pickle.load(file)

        print("Classifying all possible {0} matches...".format(len(dfr)), end='\n\n')

        # Use classifier to obtain a binary prediction or likelihood-based
        # predictions for each phrase/paragraph
        if metric in ['n_employees', 'ceo_pay_ratio']:
            # get page with high likelihood and most occurances.
            # only possible with MultinomialNB
            dfr['likelihood'] = classifier.predict_proba(dfr['text'])[:,1]
            dfr['predicted_text_class'] = predict_best_page(dfr)
        elif metric == 'healthcare':
           # Using functionality from the SpaCy classifier:
               # Classify matches and reformat dataframe,
            dfr["predicted_text_class"]=dfr["text"].apply(lambda x:
                                                           spacy_predict(x, classifier))
            dfr['probability_text_class']=dfr["text"].apply(lambda x:
                                                         spacy_probabilities(x, classifier))
        elif metric in ['ltifr', 'trifr']:
            # Get page with most phrase occurances
            dfr['predicted_text_class'] = page_highest_occurance(dfr)
        else: # all other metrics passed through the .predict() function:
            # n_contractors, n_fatalities, company_ethnic_diversity, board_ethnic_diversity,
            # international_diversity, company_gender_diversity
            # board_gender_diversity, parental_care
            # get all pages with >.5 likelihood
            dfr['predicted_text_class']=classifier.predict(dfr['text'])

        # create generic data frame columns
        dfr['class']=max(dfr['predicted_text_class'])
        dfr.loc[dfr['predicted_text_class'] == 1, 'predicted_page'] = dfr['page']
        dfr.loc[dfr['predicted_text_class'] == 1, 'predicted_text'] = dfr['text']
        dfr['predicted_page'] = dfr['predicted_page'].fillna(0).astype('Int64')
        dfr['predicted_text'] = dfr['predicted_text'].fillna('').astype(str)


        # if there is no phrase classified as positive return a dataframe without
        # values.
        if dfr['class'].all() == 0:
            df_final=pd.DataFrame({"filename":[company_info['filename']],
                                   "name":[company_info['name']],
                                   "year":[company_info['year']],
                                   "class":[0],"predicted_page":[[]],
                                   "predicted_text":[[]]})
        else:
            dfr = dfr.drop(dfr[(dfr['predicted_page'] == 0)].index, axis=0)
            df_final = dfr[['filename','name','year','class','predicted_page',
                           'predicted_text']]
            # group by documents thereby merging all page numbers into a list
            #  and also merges text strings together into a list
            df_final = df_final.groupby(['filename','name','year','class']).agg(lambda x: list(x)).reset_index()

    return df_final

def assess_metric_in_directory_list(directory_list, metric):
    """
    This function takes a list of directories containing PDF files a
    nd returns one binary classifier dataframe indicating the presence
    or absence of the specified metric within each PDF in the directories.

    --------
    Parameters:
        directory_list: list of filepaths
        metric: "n_employees" "gender_diversity", "ethnic_diversity" etc ...
    *** NOTE: metric must already be implemented in preprocessing.py ***
    RETURN
        dataframe
    ---------------
    By JET 28/08/21
    """
    dfs = pd.DataFrame()
    for directory in directory_list:
        print(directory)
        for filename in os.listdir(directory):
            print("Analysing file: ",filename)
            if filename.endswith('.pdf'):
                path = directory + '/' + filename
                dfr = extract_given_metric(path, metric)
                dfs = dfs.append(dfr)

    dfr = dfs.drop(columns=['index'], errors='ignore').reset_index()

    return dfr
