import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

IN_FILE = 'csv/Animal_Data.csv'
OUT_FILE = 'csv/Animal_Data_Cleaned.csv'

try:
    df = pd.read_csv(IN_FILE, encoding='utf-8-sig')
except FileNotFoundError:
    print(f"{IN_FILE} not found.")
    exit()

lemmatizer = WordNetLemmatizer()
base_stopwords = set(stopwords.words('english'))

#========= Custom Stopwords for Animal Data ==========
custom_stopwords = {
    'also', 'known', 'called', 'found', 'live', 'lives', 'lived',
    'one', 'two', 'three', 'many', 'most', 'some', 'often', 'usually',
    'however', 'although', 'typically', 'generally', 'commonly',
    'including', 'include', 'such', 'like', 'well', 'due',
    'animal', 'animals', 'species', 'creature', 'creatures',
    'world', 'area', 'areas', 'place', 'places', 'region', 'regions',
    'type', 'types', 'form', 'forms', 'group', 'groups',
    'year', 'years', 'time', 'times', 'day', 'days',
    'make', 'made', 'use', 'used', 'using', 'able', 'new' ,'show', 'less', 'more',
    'click', 'location', 'explore', 'map', 'show', 'native', 'origin'
}

all_stopwords = base_stopwords.union(custom_stopwords)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    cleaned = []
    for token in tokens:
        if token not in all_stopwords and len(token) > 2:
            lemma = lemmatizer.lemmatize(token)
            cleaned.append(lemma)
    return ' '.join(cleaned)

def clean_name(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    cleaned = []
    for token in tokens:
        if token not in base_stopwords and len(token) > 1:
            lemma = lemmatizer.lemmatize(token)
            cleaned.append(lemma)
    return ' '.join(cleaned)

print("Cleaning animal data...")

df['Cleaned Name']           = df['Animal Name'].apply(clean_name)
df['Cleaned Description']    = df['Description'].apply(clean_text)
df['Cleaned Habitat']        = df['Habitat'].apply(clean_text)
df['Cleaned Diet']           = df['Diet'].apply(clean_text)
df['Cleaned Behavior']       = df['Behavior'].apply(clean_text)
df['Cleaned Physical']       = df['Physical Traits'].apply(clean_text)
df['Cleaned Classification'] = df['Classification'].apply(clean_text)

df_cleaned = df.drop(columns=['Description', 'Habitat', 'Diet', 'Behavior', 'Physical Traits', 'Classification'])
df_cleaned.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
print(f"Cleaning completed. Saved to {OUT_FILE}.")
