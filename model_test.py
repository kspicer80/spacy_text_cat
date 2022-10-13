import json
import spacy
nlp = spacy.load("output/model-best")

with open('./texts_for_testing/1880_90030.json', encoding="utf-8") as f:
    test_text = json.load(f)
opinion_of_text = test_text['html_lawbox']
print(opinion_of_text)

spacy_doc = nlp(opinion_of_text)
print