import json
import pandas as ps
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

titles = dict()
headers = dict()
sub_headers = dict()
plain_text = dict()

def add_tolist_onto_dict(some_dictionary: dict, key, value):
    if key in some_dictionary:
        some_dictionary[key].append(value)
    else:
        some_dictionary[key] = [value]

def read_dataset() -> ps.DataFrame:
    with open('data/new_headers.json', 'r') as in_file:
        json_data = json.load(in_file)
    
    def print_diction(some_dict: dict):
        print(json.dumps(some_dict, indent=4, sort_keys=True))

    for filename, file_data in json_data['files'].items():
        add_tolist_onto_dict(titles, filename, file_data['title'])
        for item in file_data['marks']:
            flag, data = item
            if flag == 'he':
                add_tolist_onto_dict(headers, filename, data)
            elif flag == 'sub-he':
                add_tolist_onto_dict(sub_headers, filename, data)
            elif flag == "plain":
                add_tolist_onto_dict(plain_text, filename, data)

    X, y = [], []

    def add_to_x_y(d: dict, flag: int):
        for _, lst in d.items():
            for item in lst:
                X.append(item)
                y.append(flag)
    
    add_to_x_y(titles, 0)
    add_to_x_y(headers, 1)
    add_to_x_y(sub_headers, 2)
    add_to_x_y(plain_text, 3)

    data = ps.DataFrame({'Text': X, 'Flag': y})
    data.Text = data.Text.apply(lambda x: x.lower())
    data.Text = data.Text.apply(lambda x: x.replace('\n', ' '))

    return data

def train_naive_bayes(dataset : ps.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(dataset.Text, dataset.Flag,
                                                    test_size=0.3, random_state=11)

    mnb_text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('mnb-clf', MultinomialNB())])

    mnb_text_clf = mnb_text_clf.fit(X_train, y_train)
    train_score = mnb_text_clf.score(X_train, y_train)
    test_score = mnb_text_clf.score(X_test, y_test)
    
    print("Multinomial Naive Bayes score = {}, test score = {}".format(train_score, test_score))

def train_sgd(dataset : ps.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(dataset.Text, dataset.Flag,
                                                    test_size=0.3, random_state=11)

    sgd_text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('sgd-clf', SGDClassifier(loss='hinge', 
                                                       penalty='l2',
                                                       alpha=1e-3, 
                                                       random_state=13))])

    sgd_text_clf = sgd_text_clf.fit(X_train, y_train)
    train_score = sgd_text_clf.fit(X_train, y_train)
    test_score = sgd_text_clf.score(X_test, y_test)
    
    print("Stochastic Gradient Descent score = {}, test score = {}".format(train_score, test_score))

def train_decision_tree_classifier(dataset : ps.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(dataset.Text, dataset.Flag,
                                                    test_size=0.3, random_state=11)

    tree_text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('knn-clf', DecisionTreeClassifier())])

    tree_text_clf = tree_text_clf.fit(X_train, y_train)
    train_score = tree_text_clf.score(X_train, y_train)
    test_score = tree_text_clf.score(X_test, y_test)

    print("Decision tree train score = {}, test score = {}".format(train_score, test_score))

def train_KNN(dataset : ps.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(dataset.Text, dataset.Flag,
                                                    test_size=0.3, random_state=11)

    knn_text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('knn-clf', KNeighborsClassifier())])

    knn_text_clf = knn_text_clf.fit(X_train, y_train)
    train_score = knn_text_clf.score(X_train, y_train)
    test_score = knn_text_clf.score(X_test, y_test)

    print("K Nearest Neighbors score = {}, test score = {}".format(train_score, test_score))

if __name__ == '__main__':
    dataset = read_dataset()
    train_naive_bayes(dataset)
    #train_sgd(dataset)
    train_decision_tree_classifier(dataset)
    train_KNN(dataset)