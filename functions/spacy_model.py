#!/usr/bin/env python3
# spacy_model.py

"""
Created on Thu Aug 26 12:21:54 2021
@author: corp\isabel_e

contains functions used in custom text classification in spaCy, 
here, using an underlying simple CNN architecture
"""

import random
import spacy
from spacy.util import minibatch, compounding


def get_textcat_pipe(nlp):
    """
    Add text categorizer to the spacy pipeline using the simple_cnn
    architecture
    """
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe( 'textcat', config = {'exclusive_classes': True, 'architecture': 'simple_cnn'})
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe('textcat')

    # Adding labels to textcat
    textcat.add_label("healthcare")     # "positive"
    textcat.add_label("no_healthcare")  # "negative"
    return textcat


def convert_df(df):
    """
    Converting the dataframe into a list of tuples
    """
    df['tuples'] = df.apply(lambda row: (row['text'], row['class']), axis = 1)
    list_of_tuples = df['tuples'].tolist()
    return list_of_tuples


def load_data(list_of_tuples, split=0.8):
    """
    load df as list of tuples and split data into train/test
    """
    train_data = list_of_tuples
    # Shuffle the data
    random.shuffle(train_data)
    texts, labels = zip(*train_data)
    # get the categories for each review
    cats = [{"healthcare": bool(y), "no_healthcare": not bool(y)} for y in labels]

    # Splitting the training and evaluation data
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


def evaluate(tokenizer, textcat, texts, cats):
    """
    evaluate function from spaCy documentation
    """
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


def train_model(dev_texts, dev_cats, train_data):
    """
    # carries out the training and gives out performance metrics
    """
    # adds textcat to nlp by calling get_textcat_pipe
    nlp = spacy.load('en_core_web_sm')
    textcat = get_textcat_pipe(nlp)

    # Disabling other components
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()

        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))

        # Performing training
        for i in range(20):  # nr of iterations
            losses = {}
            batches = minibatch(train_data, size = compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd = optimizer, drop = 0.2, losses = losses)

            # Calling the evaluate() function and printing the scores
            with textcat.model.use_params(optimizer.averages):
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)

        print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'
              .format(losses['textcat'], scores['textcat_p'],
                      scores['textcat_r'], scores['textcat_f']))


def spacy_predict(inx, nlp):
    """
    loops over matches and classify those with a probablity of >= 0.7
    as positive
    """
    match_instance = nlp(inx).cats["healthcare"]

    if match_instance >= 0.7:
        return 1
    else:
        return 0


def spacy_probabilities(inx, nlp):
    """
    function that adds rounded probabilities of those instances classed as
    positive to the data frame
    """
    match_instance = nlp(inx).cats["healthcare"]
    rounded_match_instance = round(match_instance, 3)
    return rounded_match_instance
