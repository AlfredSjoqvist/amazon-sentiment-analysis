import pickle
import pandas as pd
import re as re
import os

# Defined list of stop words
stop_words = ["im", 'have', 'my', 'or', 'be', 'very', 'needn', 'his', 'shan', 'o', 'through', 
              'up', 'themselves', 'same', 'i', 'further', 'theirs', "youd", 'between', 'during', 
              "youve", 'couldn', 'myself', 'her', 'doesn', 'do', 'wasn', 'just', 'now', 'about', 
              'before', "werent", 'aren', 'himself', 'does', 'am', 'can', 'hadn', "shouldnt", 
              "hasnt", 'we', 'hasn', "shouldve", 't', 'should', "its", 'on', 'not', 'been', 
              "arent", 'which', 'because', 'having', 'until', 'once', 'most', "thatll", 'if', 
              "dont", "havent", 'won', 'weren', "couldnt", 'their', 'our', 'did', 'such', 'yours', 
              'yourselves', "isnt", 'then', 'other', 'both', 'who', 'where', "shes", 'at', "neednt", 
              'didn', 'haven', 'they', 'this', 'so', 'with', 'why', 'only', 'doing', 'few', 'd', 
              'its', 'were', 'him', 'ain', 'itself', 's', 'off', "mightnt", "wouldnt", 'what', 'a', 
              'by', "doesnt", 'the', 'ourselves', 'them', 'you', 'that', 'to', 'an', 'above', 
              'there', 'don', 'yourself', 'll', 'wouldn', 'some', 'had', 'isn', 'more', 'herself', 
              "youre", 'it', 'mustn', "wasnt", 'being', 'down', 'own', 'as', 'those', 'was', 
              'mightn', 'how', 'of', 'after', "hadnt", 'me', "didnt", 'are', 'into', 'hers', 'from', 
              'ma', "mustnt", 'any', 'no', 'ours', 'she', 'whom', 'too', 'has', "shant", 'all', 
              'here', 'these', "youll", 'is', 'out', 'and', 'nor', "wont", 'in', 'when', 'while', 
              'against', 'than', 'below', 'm', 'over', 'for', 'under', 'will', 'shouldn', 've', 
              'your', 'each', 're', 'y', 'again', 'he', 'but']

# Unpickle the data
with open(os.path.join("preprocessing", 'no_whitespaces_data.pickle'), 'rb') as handle:
    df = pickle.load(handle)


# Remove stop words
def remove_stop(sentence):
    clean_sent = []
    
    for word in sentence:
        if not word in stop_words:
            clean_sent.append(word)
    
    return clean_sent

df['Text']=df['Text'].apply(remove_stop)


# Pickle the data
with open(os.path.join("preprocessing", 'no_stop_words_data.pickle'), 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)