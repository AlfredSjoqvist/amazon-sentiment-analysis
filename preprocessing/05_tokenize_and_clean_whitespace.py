import pickle
import pandas as pd
import re as re
import os

# Unpickle the data
with open(os.path.join("preprocessing", 'lowercase_data.pickle'), 'rb') as handle:
    df = pickle.load(handle)


# Split the reviews into lists with each word being one element
df['Text']=df['Text'].apply(lambda cw : cw.split(" "))

# Remove duplicate whitespaces
def cleanempty(sentence):
    clean_sent = []
    
    for word in sentence:
        if word != "":
            clean_sent.append(word)
    
    return clean_sent

df['Text']=df['Text'].apply(cleanempty)


# Pickle the data
with open(os.path.join("preprocessing", 'no_whitespaces_data.pickle'), 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)