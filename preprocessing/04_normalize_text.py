import pickle
import pandas as pd
import re as re
import os

# Unpickle the data
with open(os.path.join("preprocessing", 'no_html_data.pickle'), 'rb') as handle:
    df = pickle.load(handle)


# Remove interpunctation
def cleanpunc(sentence):
    for illegal_character in ["?", "!", "'", '"', "#", ".", ",", ")", "(", "/"]:
        if illegal_character == "'":
            sentence = sentence.replace(illegal_character, "")
        else:
            sentence = sentence.replace(illegal_character, " ")
    return sentence
df['Text']=df['Text'].apply(lambda cw : cleanpunc(cw))

# Remove uppercase letters
df['Text']=df['Text'].apply(str.lower)


# Pickle the data
with open(os.path.join("preprocessing", 'lowercase_data.pickle'), 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)