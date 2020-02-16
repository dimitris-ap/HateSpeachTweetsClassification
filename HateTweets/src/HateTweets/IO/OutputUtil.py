from termcolor import colored
import HateTweets.IO.InputManager as InputManager
import matplotlib.pyplot as plt


LOGGING = True
LOGGING_DF = True


def log(message):
    if LOGGING:
        print(colored(message, 'blue'))


def logDF(message):
    if LOGGING_DF:
        log(message)


def save2csv(df, file_name):
    log("Exporting csv...")
    df.to_csv(InputManager.DATASET_DIR + file_name + ".csv", sep=',')
    log("Saved successfully!")


def plot(df, column):
    fig = plt.figure(figsize=(8, 6))
    df.groupby(column)[column].count().plot.bar(ylim=0)
    plt.show()
