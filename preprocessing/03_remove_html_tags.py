import pickle
import pandas as pd
import re as re
import os

# Unpickle the data
with open(os.path.join("preprocessing", 'unlabeled_data.pickle'), 'rb') as handle:
    df = pickle.load(handle)

# Remove HTML Tags
def remove_tags(string):
    result = re.sub('<.*?>',' ',string)
    return result

df['Text']=df['Text'].apply(lambda cw : remove_tags(cw))

# Pickle the data
with open(os.path.join("preprocessing", 'no_html_data.pickle'), 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)