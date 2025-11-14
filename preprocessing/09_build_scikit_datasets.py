import pickle
import pandas as pd
import re as re
import os

# Unpickle the data
with open(os.path.join("preprocessing", 'joined_sentences_data.pickle'), 'rb') as handle:
    df = pickle.load(handle)

# Put the reviews and corresponding scores in lists
review_list = []
label_list = []




def savereview(sentence):
    review_list.append(sentence)

def savescore(score):
    label_list.append(score)
    

df['Text']=df['Text'].apply(savereview)
df['Score']=df['Score'].apply(savescore)


# Pickle the data
with open(os.path.join("preprocessing", 'reviews.pickle'), 'wb') as handle:
    pickle.dump(review_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join("preprocessing", 'labels.pickle'), 'wb') as handle:
    pickle.dump(label_list, handle, protocol=pickle.HIGHEST_PROTOCOL)