import pickle
import pandas as pd
import re as re
import os
from nltk.stem.snowball import SnowballStemmer

# Unpickle the data
with open(os.path.join("preprocessing", 'no_stop_words_data.pickle'), 'rb') as handle:
    df = pickle.load(handle)
    
# Stem words   asdasd 
stemmer = SnowballStemmer("english")
df['Text'] = df['Text'].apply(lambda x: [stemmer.stem(y) for y in x])

# Pickle the data
with open(os.path.join("preprocessing", 'df_stemmed.pickle'), 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)