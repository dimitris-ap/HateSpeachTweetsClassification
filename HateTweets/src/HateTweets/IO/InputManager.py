from HateTweets.IO.OutputUtil import log, logDF
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 500)

DATASET_DIR = "../../resources/"
DATASET_FILENAME = "hate_tweets.csv"
# DATASET_FILENAME = "preprocessed.csv"
# DATASET_FILENAME = "testing.csv"


def load_data() -> pd.DataFrame:
    log("Reading " + DATASET_FILENAME)
    df = pd.read_csv(DATASET_DIR + DATASET_FILENAME, index_col=0)
    logDF(df.head())
    # logDF(df.info())
    return df


