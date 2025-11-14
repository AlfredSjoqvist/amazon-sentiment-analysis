import pickle
import pandas as pd
import re as re
import os

# Unpickle the data
with open(os.path.join("preprocessing", 'df_stemmed.pickle'), 'rb') as handle:
    df = pickle.load(handle)


# Join the list with words around a whitespace
df['Text']=df['Text'].apply(lambda cw : " ".join(cw))


# Pickle the data
with open(os.path.join("preprocessing", 'joined_sentences_data.pickle'), 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)