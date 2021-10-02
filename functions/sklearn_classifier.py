# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import string
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score,f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV

# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from spacy.lang.en import English

parser = English()

SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]


#funcitons
def cleanup_text(docs, logging=False):
    """ This function takes a doc-object and does basic text cleanup
    like .lower().strip(), lemma, tokenisation, stopword and punctuation
    extraction"""
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(str(doc), disable=['parser', 'ner'])

        # tokenise text and remove stopwords
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]

        #join tokens together
        tokens = ' '.join(tokens)
        #add sections
        texts.append(tokens)

        # return as a df
    return pd.Series(texts)


class CleanTextTransformer(TransformerMixin):
    """Add the basic text cleaning function into a class"""
    def transform(self, X, **transform_params):
        # return the cleaned text
        return [cleanText(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self

def get_params(self, deep=True):
    return {}

def cleanText(text):
    """ Get rid of symbols generated through text parsing"""
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text

def tokenizeText(sample):
    """ Tokenise text that is not a stopword or symbol"""
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in stopwords.words('english')]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    return tokens

def printNMostInformative(vectorizer, clf, N, component=0):
    """ extract feature names from vectoriser and apply
    weights from the classifier to obtain the most predictive
    words used in classification """
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[component], feature_names))
    topclass1 = coefs_with_fns[:N]
    topclass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topclass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topclass2:
        print(feat)


def train_NLP_classifier(dfr,ngram_range = (1,1),classifier = 'LinearSVC',
                         balance_classes = False, C = 1, alpha = 1):
    """ This function will take the dataframe and will ONLY train a classifier
        over text strings, feeding back the most predictive features.

       Parameters
       ----------
       dfr : dataframe
               dataframe with a column 'text' with strings
               and a column binary variable 'class'
       ngram_range    : tuple
               2x1 tuple, change the second value to increase
               the number of words the classifier can use as
               features. (1,1) = one word; (1,2) = one word +
               pairs of words; etc.

               The lower and upper boundary of the range of
               n-values for different word n-grams or char
               n-grams to be extracted.
       classifier  : string
               'SGDClassifier', 'MultinomialNB', or the
               default 'LinearSVC'
       balance_classes : Boolean
               Set True to balance the two classes to match the negative and
               the positive cases.
       C : float
               C parameter for the LinearSVC
       alpha : float
               alpha parameter for the SGDClassifier and MultinomialNB.

       Returns
       ----------
       pipe : sklearn.model
               Pre-trained pipeline for classification.


       By Jasper Hajonides (27/08/2021)
       """
    if not isinstance(dfr, pd.DataFrame):
        raise ValueError("input a panda dataframe with columns 'text' and 'class'.")

    # convert to lists
    train = dfr['text'].tolist()
    labelstrain = dfr['class'].tolist()

    # convert strings to tokens and vectorise
    vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=ngram_range)

    # define classifier
    if classifier == 'SGDClassifier':
        clf = SGDClassifier(loss='hinge', penalty='l2', alpha=alpha,
                            random_state=42, max_iter=5, tol=None)
    elif classifier == 'MultinomialNB':
        clf = MultinomialNB(alpha=alpha)
    else:
        clf = LinearSVC(C=C)

    # Merge all previous steps into a single pipeline
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer),
                     ('TermFreq',TfidfTransformer(use_idf=False)),('clf', clf)])

    # train
    pipe.fit(train, labelstrain)

    return pipe


def classify_NLP_df(df_in, ngram_range = (1,1),classifier = 'LinearSVC',
                    balance_classes = False, C = 1, alpha = 1,
                    replace_nums=False):
    """ This function will take the dataframe and will run
        classification over text strings, feeding back the
        most predictive features.
        Parameters
        ----------
        dfr : dataframe
                dataframe with a column 'text' with strings
                and a column binary variable 'class'
        ngram_range    : tuple
                2x1 tuple, change the second value to increase
                the number of words the classifier can use as
                features. (1,1) = one word; (1,2) = one word +
                pairs of words; etc.

                The lower and upper boundary of the range of
                n-values for different word n-grams or char
                n-grams to be extracted.
        classifier  : string
                'SGDClassifier', 'MultinomialNB', or the
                default 'LinearSVC'
        balance_classes : Boolean
                Set True to balance the two classes to match the negative and
                the positive cases. Set "1:3" to acheive a positive:negative
                case ratio of 1:3
        C : float
                C parameter for the LinearSVC
        alpha : float
                alpha parameter for the SGDClassifier and MultinomialNB.

        replace_nums : Boolean
                 Set True to mark dates with "YEAR", and all remaining digits
                 with "NUM" placeholders in within dataframe["text"]

        Returns
        ----------
        pipe : sklearn.model
                Pre-trained pipeline for classification.
        output : dictionary
            pred: ndarray
                binary class predictions for every text string.
            pred_proba: ndarray
                class likelihood predictions for every text string
            accuracy: float
                accuracy of classifier by comparing class predictions
                with predefined labels in dfr['class']
            precision: float
                precision of model
            recall: float
                recall rate for model
            f1: float
                f1 score

    By Jasper Hajonides (16/08/2021)
        """

    # subsample an equal number of cases for both classes.
    unequal_classes = (df_in['class'].sum() < (len(df_in)-df_in['class'].sum()))
    if unequal_classes is True and balance_classes is True:
        print('positive {} and negative {}, subsampling to {}/{} cases for each...'.format(
            df_in['class'].sum(), len(df_in)-df_in['class'].sum(),
            df_in['class'].sum(),df_in['class'].sum()))
        df_minority = df_in.loc[df_in['class']==1,:]
        df_majority = df_in.loc[df_in['class']==0,:]
        df_majority_downsampled = resample(df_majority,replace=False,
                                           n_samples=df_in['class'].sum()) #,random_state=123
        dfr = pd.concat([df_majority_downsampled, df_minority]).reset_index()

    # subsample at a 1:3 true:false ratio.
    elif unequal_classes is True and balance_classes is "1:3":
        print('positive {} and negative {}, subsampling to {}/{} cases for each...'.format(
            df_in['class'].sum(), len(df_in)-df_in['class'].sum(),
            df_in['class'].sum(),3*df_in['class'].sum()))
        df_minority = df_in.loc[df_in['class']==1,:]
        df_majority = df_in.loc[df_in['class']==0,:]
        df_majority_downsampled = resample(df_majority,replace=False,
                                           n_samples=3*df_in['class'].sum()) #,random_state=123
        dfr = pd.concat([df_majority_downsampled, df_minority]).reset_index()
    else:
        dfr = df_in
    # initialise empty array
    output = {}
    output['preds'] = np.zeros(len(dfr['class']))
    output['likelihood'] = np.zeros((len(dfr['class']),2))

    # Replace any digits in text with either "YEAR" or "NUM", depending on format
    if replace_nums is True:
        dfr['text'] = dfr['text'].replace(r"^(19|20)\d{2}$", "YEAR", regex=True)
        dfr['text'] = dfr['text'].replace(r"\d+(\.\d+)?(\%)?", "NUM", regex=True)

    # cross validation, splitting dataframe in train and test sets
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=42)
    for train_index, test_index in rskf.split(dfr['text'], dfr['class']):
        x_train, x_test = dfr['text'][train_index], dfr['text'][test_index]
        y_train  = dfr['class'][train_index]

        # convert to lists
        train = x_train.tolist()
        labelstrain = y_train.tolist()
        test = x_test.tolist()

        # convert strings to tokens and vectorise
        vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=ngram_range)

        # define classifier
        if classifier == 'SGDClassifier':
            clf = SGDClassifier(loss='hinge', penalty='l2', alpha=alpha,
                                random_state=42, max_iter=5, tol=None) #SGDClassifier() #LinearSVC()
        elif classifier == 'MultinomialNB':
            clf = MultinomialNB(alpha=alpha)
        else:
            clf = LinearSVC(C=C)

        # Merge all previous steps into a single pipeline
        pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer),
                         ('TermFreq',TfidfTransformer(use_idf=False)),('clf', clf)])

        # train
        pipe.fit(train, labelstrain)

        # test
        output['preds'][test_index] = pipe.predict(test)

        # If the multinomialNB classifier is used we also obtain likelihood
        # predictions and store them in the dictionary. Other classifiers
        # currently dont support this function.
        if classifier == 'MultinomialNB':
            output['likelihood'][test_index,:] = pipe.predict_proba(test)

    # print and put scores in dictionary
    output['accuracy'] = accuracy_score(dfr['class'],output['preds'])
    print("accuracy:", output['accuracy'])
    output['precision'] = precision_score(dfr['class'],output['preds'])
    print("precision:", output['precision'])
    output['recall'] = recall_score(dfr['class'],output['preds'])
    print("recall:", output['recall'])
    output['f1'] = f1_score(dfr['class'],output['preds'])
    print("F1 score:", output['f1'])

    # next, we obtain and print out the most predictive words used by the classifier
    print("Top ten features that predict {} report vs {}: ".format(dfr['class'].unique()[0],
                                                                   dfr['class'].unique()[1]))

    # Using function to get words that are most informative for classifier
    printNMostInformative(vectorizer, clf, 10, component=0)

    return pipe, output


def classify_NLP_grid_search(df_in, balance_classes=False, replace_nums=False):
    """ This function will take the dataframe and will run
        classification over text strings, feeding back the
        most predictive features.

        Parameters
        ----------
        dfr : dataframe
                dataframe with a column 'text' with strings
                and a column 'report' with a class identfier
        ngram_range    : tuple
                2x1 tuple, change the second value to increase
                the number of words the classifier can use as
                features. (1,1) = one word; (1,2) = one word +
                pairs of words; etc.

                The lower and upper boundary of the range of
                n-values for different word n-grams or char
                n-grams to be extracted.
        classifier  : string
                'SGDClassifier', 'MultinomialNB', or the
                default 'LinearSVC'
        balance_classes : Boolean
                Set True to balance the two classes to match the negative and
                the positive cases.
                Set "1:3" to acheive a positive:negative case ratio of 1:3
        C : float
                C parameter for the LinearSVC
        alpha : float
                alpha parameter for the SGDClassifier and MultinomialNB.
        replace_nums : Boolean
                 Set True to mark dates with "YEAR", and all remaining digits
                 with "NUM" placeholders in within dataframe["text"]
        Return
        ----------
        grid : sklearn.model
                Pre-trained pipeline for classification.   
    By Jasper Hajonides (20/08/2021)
        """

    # subsample an equal number of cases for both classes.
    unequal_classes = (df_in['class'].sum() < (len(df_in)-df_in['class'].sum()))
    if unequal_classes is True and balance_classes is True:
        print('positive {} and negative {}, subsampling to {}/{} cases for each...'.format(
            df_in['class'].sum(), len(df_in)-df_in['class'].sum(),
            df_in['class'].sum(),df_in['class'].sum()))
        df_minority = df_in.loc[df_in['class']==1,:]
        df_majority = df_in.loc[df_in['class']==0,:]
        df_majority_downsampled = resample(df_majority,replace=False,
                                           n_samples=df_in['class'].sum())
        dfr = pd.concat([df_majority_downsampled, df_minority]).reset_index()

    # subsample at a 1:3 true:false ratio.
    elif unequal_classes is True and balance_classes is "1:3":
        print('positive {} and negative {}, subsampling to {}/{} cases for each...'.format(
            df_in['class'].sum(), len(df_in)-df_in['class'].sum(),
            df_in['class'].sum(),3*df_in['class'].sum()))
        df_minority = df_in.loc[df_in['class']==1,:]
        df_majority = df_in.loc[df_in['class']==0,:]
        df_majority_downsampled = resample(df_majority,replace=False,
                                           n_samples=3*df_in['class'].sum())
        dfr = pd.concat([df_majority_downsampled, df_minority]).reset_index()
    else:
        dfr = df_in

    # Replace any digits in text with either "YEAR" or "NUM", depending on format
    if replace_nums is True:
        dfr['text'] = dfr['text'].replace(r"\d+(\.\d+)?(\%)?", "NUM", regex=True)

    x_data = dfr['text']
    y_data = dfr['class']

    # convert strings to tokens and vectorise
    vectorizer = CountVectorizer(tokenizer=tokenizeText) #, ngram_range=ngram_range

    # define all classifiers
    lin_svc = LinearSVC()
    sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,
                        max_iter=5, tol=None) #SGDClassifier() #LinearSVC()
    mnb = MultinomialNB()

    # Ensemble classifier definition
    ensemble_clf=[lin_svc, sgd, mnb]
    ensemble_names = ['Linear Support Vector Classifier', 'Stochastic gradient descent Classifier',
                   'Multinominal Naive Bayes']

    # define parameters for each estimator/classifier, can be extended
    params1 = {"clf__C":[0.001, .01, .1, 1, 10, 100]}
    params2 = {"clf__alpha": [1e-3, 1e-2, 1e-1, 1, 10]}
    params3 = {"clf__alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100]}
    parameters_list=[params1, params2, params3]

    # Run classification for each classifier
    for i in range(len(ensemble_clf)):

        # Merge all previous steps into a single pipeline
        pipe = Pipeline([('cleanText', CleanTextTransformer()),
                         ('vectorizer', vectorizer),
                         ('TermFreq',TfidfTransformer(use_idf=False)),
                         ('clf', ensemble_clf[i])])

        # perform hyperparameter tuning
        parameters_list[i]['vectorizer__ngram_range'] = [(1,1),(1,2),(1,3)]
        grid = GridSearchCV(pipe, param_grid=parameters_list[i],scoring='f1')

        # fit grid with parameters and print the best set of parameters + score
        grid.fit(x_data,y_data).best_estimator_
        print(grid.best_params_)
        print(ensemble_names[i])
        print('best F1-score', grid.best_score_)

        # return the best classifier 
        if i == 0:
            print('Using ' + ensemble_names[i] + ' as best classifier')
            best_grid = grid
        elif grid.best_score_ > best_grid.best_score_:
            print('Outputting ' + ensemble_names[i] + ' as new best classifier')
            best_grid = grid

    return grid
