import pickle
import pandas as pd
import re as re
import os

# Unpickle the data
with open(os.path.join("data", 'raw_dataframe.pickle'), 'rb') as handle:
    df = pickle.load(handle)


# Remove data labels we wont use
useless_labels = ["Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator", "Time", "Summary"]

for label in useless_labels:
    df.drop(label, inplace=True, axis=1)

# Remove duplicates
df.drop_duplicates(inplace=True)


# Pickle the data
with open(os.path.join("data", 'unlabeled_data.pickle'), 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)