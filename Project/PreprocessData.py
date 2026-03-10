import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# ========== PATH CONFIG ==========
PROJECT_DIR    = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_PATH = os.path.normpath(os.path.join(PROJECT_DIR, '..', 'nltk_data'))
CSV_DIR        = os.path.join(PROJECT_DIR, 'csv')
IN_FILE        = os.path.join(CSV_DIR, 'Animal_Data.csv')
OUT_FILE       = os.path.join(CSV_DIR, 'Animal_Data_Cleaned.csv')
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
# ==================================

nltk.data.path.insert(0, NLTK_DATA_PATH)
nltk.download('punkt',     download_dir=NLTK_DATA_PATH)
nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH)
nltk.download('stopwords', download_dir=NLTK_DATA_PATH)
nltk.download('wordnet',   download_dir=NLTK_DATA_PATH)

try:
    df = pd.read_csv(IN_FILE, encoding='utf-8-sig')
except FileNotFoundError:
    print(f"{IN_FILE} not found.")
    exit()

lemmatizer     = WordNetLemmatizer()
base_stopwords = set(stopwords.words('english'))

# ========= Custom Stopwords for Animal Data ==========
custom_stopwords = {

    # --- Web / UI artifacts (scraped noise) ---
    'click', 'explore', 'map', 'show', 'read', 'learn', 'discover',
    'visit', 'page', 'article', 'website', 'link', 'photo', 'image',
    'copyright', 'reserved', 'right', 'source', 'reference', 'fact',

    # --- Filler adverbs & conjunctions ---
    'also', 'however', 'although', 'though', 'typically', 'generally',
    'commonly', 'usually', 'often', 'sometimes', 'occasionally',
    'therefore', 'thus', 'hence', 'furthermore', 'moreover',
    'additionally', 'nevertheless', 'nonetheless',

    # --- Generic verbs (low discriminative value) ---
    'make', 'made', 'use', 'used', 'using', 'able',
    'include', 'including', 'become', 'consider', 'get', 'give',
    'take', 'keep', 'let', 'put', 'seem', 'tell', 'try', 'mean',
    'call', 'called', 'known', 'refer',

    # --- Overly generic nouns (appear in nearly every document) ---
    'animal', 'creature', 'organism', 'thing', 'example',
    'type', 'form', 'way', 'part', 'number', 'member',
    'world', 'nature', 'environment',

    # --- Vague quantifiers (not useful for retrieval) ---
    'many', 'most', 'some', 'several', 'various', 'certain',
    'few', 'less', 'more', 'much', 'lot', 'enough',

    # --- Time references (rarely discriminative for animals) ---
    'year', 'month', 'day', 'hour', 'time', 'period',
    'season', 'century', 'age',

    # --- Filler preposition-like phrases ---
    'due', 'such', 'well', 'new', 'old', 'based',

}
all_stopwords = base_stopwords.union(custom_stopwords)

def clean_text(text):
    text    = str(text).lower()
    text    = re.sub(r'[^a-z\s]', ' ', text)
    tokens  = word_tokenize(text)
    cleaned = []
    for token in tokens:
        if token not in all_stopwords and len(token) > 2:
            lemma = lemmatizer.lemmatize(token)
            cleaned.append(lemma)
    return ' '.join(cleaned)

def clean_name(text):
    text    = str(text).lower()
    text    = re.sub(r'[^a-z\s]', ' ', text)
    tokens  = word_tokenize(text)
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