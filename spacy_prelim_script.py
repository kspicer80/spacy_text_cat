import pandas as pd
from typing import Set, List, Tuple
import spacy
from spacy.tokens import DocBin
nlp = spacy.load('en_core_web_lg')
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def make_docs(data: List[Tuple[str, str]], target_file: str, cats: Set[str]):
    docs = DocBin()
    # Use nlp.pipe to efficiently process a large number of text inputs,
    # the as_tuple arguments enables giving a list of tuples as input and
    # reuse it in the loop, here for the labels
    for doc, label in tqdm(nlp.pipe(data, as_tuples=True, disable=['tagger', 'parser', 'attribute_ruler', 'lemmatizer']), total=len(data)):
        # Encode the labels (assign 1 to the label)
        for cat in cats:
            doc.cats[cat] = 1 if cat == label else 0
        docs.add(doc)
    docs.to_disk(target_file)
    return docs

def make_docs_01(data):
    docs = []

    for doc, label in tqdm(nlp.pipe(data, as_tuples=True), total = len(data)):
        for cat in cats:
            doc.cats[cat] = 1 if cat == label else 0
        docs.append(doc)
    return docs

df = pd.read_json('training_json_file.json', orient='records', encoding='utf-8')
print(df.head())
df['text'] = df['cleaned_html'].replace(r'\n',' ', regex=True)
df['label'] = df['label'].astype('str')
resampled_df = df.groupby('label').apply(lambda x: x.sample(1000)).reset_index(drop=True)
#print(resampled_df.head())
#print(resampled_df['label'].value_counts())

cats = df.label.unique().tolist()
print(cats)


X_train, X_valid, y_train, y_valid = train_test_split(df["text"].values, df["label"].values, test_size=0.2)

tqdm(make_docs(list(zip(X_train, y_train)), "train_full.spacy", cats=cats))
tqdm(make_docs(list(zip(X_valid, y_valid)), "valid_full.spacy", cats=cats))
print("Finished making all the docs!")

