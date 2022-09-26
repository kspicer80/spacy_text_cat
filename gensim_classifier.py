import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import CountVectorizer
warnings.filterwarnings('ignore')
 
# Reading the data
file = '/Users/kspicer/Desktop/spacy_textcat/cleaned_dataframe.jsonl'
df = pd.read_json(file, lines=True)
 
# Printing number of rows and columns
print(df.shape)

corpus = df['cleaned_text'].values()
vectorizer = CountVectorizer(corpus)
X = vectorizer.fit_transform(corpus)

CountVectorizedData = pd.DataFrame(X.to_array(), columns=vectorizer.get_feature_names())
CountVectorizedData['label'] = df['label']


