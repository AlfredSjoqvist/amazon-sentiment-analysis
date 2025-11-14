import pickle
import pandas as pd
import re as re
import os

# Read data frame from CSV
df = pd.read_csv(os.path.join("preprocessing", "Reviews.csv"))

# Save the data frame to a pickle file for later use
with open(os.path.join("preprocessing", 'raw_dataframe.pickle'), 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)