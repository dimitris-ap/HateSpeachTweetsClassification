from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from HateTweets.IO.OutputUtil import log
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import make_pipeline


def classify(df):
    X = df['tweet']
    y = df['class']

    run('NaiveBayes', X, y)
    run('RandomForest', X, y)
    run('LinearSVC', X, y)


def run(classifier, X, y):
    log("Running " + classifier + " Classifier...")

    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = make_pipeline(TfidfVectorizer(), get_classifier(classifier))
    # model = make_pipeline(TfidfVectorizer(), SMOTE(random_state=42), get_classifier(classifier))

    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)

    calc_and_print_metrics(y_test, y_predicted)
    confusion_matrix(y_test, y_predicted, classifier)


def get_classifier(classifier):
    if classifier == 'NaiveBayes':
        return MultinomialNB(alpha=0.1)
    elif classifier == 'RandomForest':
        return RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0, class_weight="balanced")
    elif classifier == 'LinearSVC':
        return LinearSVC()


def calc_and_print_metrics(y_test, y_predicted):
    accuracy = metrics.accuracy_score(y_test, y_predicted)
    recall = metrics.recall_score(y_test, y_predicted, average='macro')
    precision = metrics.precision_score(y_test, y_predicted, average='macro')
    f1 = metrics.f1_score(y_test, y_predicted, average='macro')

    print("Accuracy: %f" % accuracy)
    print("Recall: %f" % recall)
    print("Precision: %f" % precision)
    print("F1: %f" % f1)


def confusion_matrix(y_test, y_predicted, classifier):
    conf_matrix = metrics.confusion_matrix(y_test, y_predicted)
    plt.figure(figsize=(18, 12))

    hm = sns.heatmap(conf_matrix.T,
                     cmap="Blues",
                     linewidths=2,
                     square=False,
                     annot=True,
                     cbar=False,
                     fmt='d',
                     xticklabels=['0', '1', '2'],
                     yticklabels=['0', '1', '2'])

    bottom, top = hm.get_ylim()
    hm.set_ylim(bottom + 0.5, top - 0.5)

    plt.title(classifier)
    plt.xlabel('True output')
    plt.ylabel('Predicted output')
    plt.show()