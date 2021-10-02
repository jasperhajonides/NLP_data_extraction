#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# preprocessing.py
"""
Contains functions to obtain company information and pre-processing
for metric extraction pipelines.
"""
import re
import spacy
import pandas as pd
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from cleaning import parse_file,toy_clean,rep_char_search

nlp = spacy.load("en_core_web_sm")
matcher=Matcher(nlp.vocab)

def define_company_dictionary(path):
    """Returns dictionary of company information

    Args:
        path(str): Company's report path

    Returns:
        company_info (dict): Dictionary with
        keys=['name', 'filename', 'type', 'year', 'clean_text']
    """
    filename = path.split('/')[-1]
    company_name = ' '.join(filename.split('_Annual_Report_')[0].split('_'))
    year = filename.split('_Annual_Report_')[1].split('.')[0]
    type_of_report = 'Annual Report'

    pdf_dict = parse_file(path)
    cleaned_text = toy_clean(pdf_dict, table_contents =True)
    cleaned_text = rep_char_search(cleaned_text)

    company_info_dict = {'name' : company_name,
                         'filename' : filename,
                         'type' : type_of_report,
                         'year' : year,
                         'clean_text': cleaned_text,
    }
    return company_info_dict

def select_pages(page_dict, metric):
    """Returns dictionary with selected pages of text containing
    keywords or expressions for a given metric

    Args:
        page_dict (dict):  Dictionary with keys=['page'] values=text_list
        metric (str or list):
            If metric is a string it loads in a predefined set of keywords.
            Desired metric current possible values:
                
                'n_employees'
               
          
            If metric is a list of strings, this list of keywords will then be
            used to select relevant pages.

    Returns:
        selected_dict: Dictionary with keys=['page'] values=text_list
                        where the pages are the ones containing
                        keywords regarding a specific metric
    """
    selected_dict ={}

    if metric == 'n_employees':
        keywords =['employ', 'team', 'workforce', 'colleague', 'staff', 'headcount']


    elif isinstance(metric, list):
        # if keywords is defined externally we use metric as a list of keywords
        keywords = metric
    else:
        raise ValueError(f'The metric {metric} is not implemented.')

    # Now use the keywords to search the document and select pages.
    page_list = []
    for page,text_list in page_dict.items():
        text = ' '.join(text_list).lower()
        for keyword in keywords:
            if (re.search(keyword,text) != None) and (page not in page_list):
                page_list.append(page)

    selected_dict={page:page_dict[page] for page in page_list}

    return selected_dict

def run_nlp(page_dict):
    """Runs nlp pipe and to selected pages of text
    dictionary.

    Args:
        page_dict (dict): Dictionary with keys=['page'] values=text_list

    Returns:
        nlp_dict (dict): Dictionary with
        keys=['page'], value = nlp_text
    """
    nlp_dict = {page:list(nlp.pipe(text_list)) for page, text_list in page_dict.items()}
    return nlp_dict

def pattern_definition(metric):
    """
    Define patterns to be used on SpaCy matchers for filtering pages in the
    document for each of the desired metrics.
    
    Args:
        metric (str): Desired metric with possible values:
            
                'n_employees'
               
                
    Returns:
        patterns (list): Patterns to feed to a SpaCy Matcher or PhraseMatcher
        object.
    """
    if metric == 'n_employees':
        employee_synonyms = ["employee",
                             "people",
                             "person",
                             "colleague",
                             "team",
                             "staff",
                             "full-time",
                             "fte"]
        pattern1=[
          {"LEMMA": {"IN": ["total", "average"]}},
          {"OP":"?"},
          {"OP":"?"},
          {"LEMMA": "number", "OP":"?"},
          {"LEMMA": "of", "OP":"+"},
          {"LEMMA": {"IN": employee_synonyms}},
        ]
        pattern2=[
          {"LEMMA": {"IN": ["employ", "hire"]}},
          {"OP":"?"},
          {"OP":"?"},
          {"OP":"?"},
          {"OP":"?"},
          {"ENT_TYPE":"CARDINAL","OP":"+"},
          {"LEMMA": {"IN": employee_synonyms}},
          {"POS":"NOUN","OP":"?"}
        ]
        pattern3 =[
            {"LEMMA": {"IN": ["the", "our", "with", "have"]}},
            {"POS": "ADP", "OP":"?"},
            {"POS": "ADP", "OP":"?"},
            {"ENT_TYPE":"CARDINAL","OP":"+"},
            {"LEMMA": {"IN": employee_synonyms + ["headcount", "workforce"]}}
        ]

        pattern4 =[
            {"LEMMA": {"IN": ["total", "average"]}},
            {"LEMMA": {"IN": employee_synonyms + ["headcount", "workforce"]}}
        ]

        pattern5 = [
            {"LEMMA":{"IN":["with","have"]}},
            {"LEMMA": {"IN":["headcount","team", "workforce", "staff"]}},
            {"LEMMA": "of"},
            {"ENT_TYPE": "CARDINAL"}
            ]

        pattern6 = [
            {"LEMMA": "there"},
            {"LEMMA": "be"},
            {"ENT_TYPE":"CARDINAL"},
            {"LEMMA": {"IN": employee_synonyms}}
            ]

        pattern7 = [
            {"ENT_TYPE": "CARDINAL"},
            {"LEMMA": {"IN": employee_synonyms}},
            {"LEMMA": "work"}
            ]

        patterns = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7]
    else:
        raise ValueError(f'The metric {metric} is not implemented.')
    return patterns

def define_matcher(patterns, matcher_type='text'):
    """Returns word matcher for a given metric.

    Args:
        patterns (list):
            ... list of lists can be obtained using pattern_definition()
            or can be defined manually.
        matcher_type (str): Takes values
          'text': defines normal Matcher instance with given patterns,
          'phrase': defines PhraseMatcher instance with given patterns

    Returns:
        if matcher_type='text'
            matcher (Matcher): Token Matcher object matching patterns
                                for the chosen metric
        if matcher_type='phrase'
            matcher (PhraseMatcher): PhraseMatcher object matching patterns
                                    for the chosen metric
    """
    if not isinstance(patterns, list):
        raise ValueError('Input should be a list of spaCy patterns.')

    if matcher_type == 'text':
        matcher = Matcher(nlp.vocab)

    elif matcher_type == 'phrase':
        matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

    else:
        raise ValueError(f'{matcher_type} is not a valid matcher type')

    # loop over patterns and add them to the matcher
    for i in range(len(patterns)):
        matcher.add("ID_pattern{}".format(i), [patterns[i]])

    return matcher

def create_match_dataframe(path, metric):
    """Creates dataframe with possible matches of the desired metric.

    Args:
        path (str): Reports path
        metric: desired metrics taking values between:
            n_employees'
               
        matcher_type (str): Takes values
          'text': defines normal Matcher instance with given patterns,
          'phrase': defines PhraseMatcher instance with given patterns

    Returns:
        match_dataframe: Dataframe with columns=['string', 'page']
        containing the matched sentences from a given metric
        and respective page number
    """
    company_info = define_company_dictionary(path)
    selected_text = select_pages(company_info['clean_text'], metric)
    nlp_text = run_nlp(selected_text)

    # Define pattern
    pattern = pattern_definition(metric)
    
    # Look for metrics that use token matcher
    if metric in ["n_employees"]:
        matcher_type="text"

    matcher = define_matcher(pattern, matcher_type)
    
    # Loop over the text outputted by the matcher to create a dataframe of
    # matches for a given metric.
    # The next step would be to apply the binary classifier of the metric
    # to the text contained in this dataframe.
    
    filenames = []
    names = []
    years = []
    match_pages = []
    match_text = []
    match_strings = []
    entity_starts =[]
    entity_ends=[]

    for page, text_list in nlp_text.items():
        for text in text_list:
            match = matcher(text)
            if len(match)>0:
                match_id, start, end = match[0]
                span = text[start:end]
                filenames.append(company_info['filename'])
                names.append(company_info['name'])
                years.append(company_info['year'])
                match_text.append(text.text)
                match_strings.append(span.text)
                match_pages.append(page)
                entity_starts.append(start)
                entity_ends.append(end)

    match_dataframe=pd.DataFrame({'filename': filenames,
                                  'name': names,
                                  'year': years,
                                  'text':match_text,
                                  'match_string': match_strings,
                                  'page':match_pages,
                                  'start': entity_starts,
                                  'end': entity_ends})
    return match_dataframe



def predict_best_page(dfr):
    """ This function finds the page with the best predictions
        for the target variable.

        input
        -----
        dfr: pd.DataFrame
            containing rows of different phrases and columns for:
            -page numbers of these phrases ('page')
            -likelihood estimations ('likelihood')

        returns
        -----
        selected_pages: list
            a binary list with index for each sentence

        Jasper Hajonides 30082021
        """
    #What page do we find the best prediction based on likelihood
    indices_dfr = dfr.loc[dfr['likelihood'] > 0.6, :].groupby('page').agg(
        page_count = ("page", 'count')).join(dfr.groupby('page').agg(
            likelihood=("likelihood", 'max')))

    # sort by likelihood
    indices_dfr = indices_dfr.reset_index().sort_values(by='likelihood',ascending=False).reset_index()
    #count ratio of occurances per page
    indices_dfr['ratio_of_occurances']=indices_dfr['page_count']/indices_dfr['page_count'].sum()
    #page with good likelihood and highest page count.
    top_index = indices_dfr.loc[(indices_dfr['likelihood'] > .6) &
                                (indices_dfr['page_count'] == indices_dfr['page_count'].max()),
                                'index']
    # if there happen to be multiple cases what predict well we take the first result.
    if len(top_index) > 0:

        # add page number if likelihood is <.4
        dfr_filt = indices_dfr.loc[(indices_dfr['likelihood'] > .6) &
                                   (indices_dfr['page_count'] == indices_dfr['page_count'].max())
                                   ].head(1).reset_index()

        convert = map(int, (dfr['page'] == int(dfr_filt['page'])) & (dfr['likelihood'] > .6))
        selected_pages = list(convert)
    else:
        selected_pages = [0]*len(dfr)

    return selected_pages


def page_highest_occurance(dfr):
    """ This function finds the page that is most frequently occuring in dataframe

        input
        -----
        dfr: pd.DataFrame
            containing rows of different phrases and columns for:
            -page numbers of these phrases ('page')

        returns
        -----
        selected_pages: list
            a binary list with index for each sentence

        Jasper Hajonides 31082021
        """
    #Get number of page occurances
    indices_dfr = dfr.groupby('page').agg(
        page_count = ("page", 'count')).reset_index()

    #page with highest page count.
    top_index = indices_dfr.loc[(indices_dfr['page_count'] == indices_dfr['page_count'].max()),
                                'page'].reset_index()

    # if there happen to be multiple cases what predict well we take the first result.
    page = list(top_index.loc[:,'page'])

    # find phrases in original data frame that occur on the most freq. occurring page
    dfr['highest_occurring_page'] = False
    for p in range(len(dfr)):
        dfr['highest_occurring_page'][p] = dfr['page'][p] in page
    convert = map(int, dfr['highest_occurring_page'] )

    return list(convert)
