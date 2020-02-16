import HateTweets.IO.InputManager as InputManager
import HateTweets.Classification as classifiers
import HateTweets.preprocess.PreprocessUtil as PreProcess
from HateTweets.IO.OutputUtil import log, logDF, save2csv, plot


log("### HATE TWEETS CLASSIFICATION ###")

df = InputManager.load_data()
df = PreProcess.run(df)

plot(df, 'class')
# save2csv(df, "preprocessed")

classifiers.classify(df)

log("##################################")
