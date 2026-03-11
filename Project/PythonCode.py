# Start of ScrapeData.py
from bs4 import BeautifulSoup
import requests
import csv
import random
import os
from urllib.parse import urljoin

# ========== PATH CONFIG ==========
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR     = os.path.join(PROJECT_DIR, 'csv')
OUT_FILE    = os.path.join(CSV_DIR, 'Animal_Data.csv')
os.makedirs(CSV_DIR, exist_ok=True)
# ==================================

# ========== CONFIG ==========
BASE_URL    = "https://a-z-animals.com"
ANIMAL_URL  = "https://a-z-animals.com/animals/"
SAMPLE_SIZE = 500
random.seed(41)
# ============================

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Referer": "https://www.google.com/",
}

session = requests.Session()
session.headers.update(headers)

EXCLUDE_KEYWORDS = [
    'animals-that-start', 'scientific', 'class', 'location','page', 'quiz', 'blog'
]

# ========== Gather Animal Links ==========
print("Fetching animal links from A-Z Animals website...")
res  = session.get(ANIMAL_URL)
soup = BeautifulSoup(res.text, 'html.parser')

all_links   = [a['href'] for a in soup.find_all('a', href=True)]
animal_pool = []

for href in all_links:
    if '/animals/' not in href:
        continue
    if any(kw in href for kw in EXCLUDE_KEYWORDS):
        continue
    clean = href.replace('https://a-z-animals.com', '').strip('/')
    parts = [p for p in clean.split('/') if p]
    if len(parts) == 2 and parts[0] == 'animals':
        animal_pool.append(href)

animal_pool = list(dict.fromkeys(animal_pool))
print(f"Total animals found: {len(animal_pool)}")

# ========== Random Sampling ==========
sampled = random.sample(animal_pool, min(SAMPLE_SIZE, len(animal_pool)))
print(f"Randomly selected {len(sampled)} animals to scrape...\n")

# ========== Scrape Data ==========
def get_text_by_keywords(soup, keywords):
    results = []
    for tag in soup.find_all(['h2', 'h3', 'h4']):
        if any(kw.lower() in tag.get_text().lower() for kw in keywords):
            next_tag = tag.find_next_sibling()
            while next_tag and next_tag.name not in ['h2', 'h3', 'h4']:
                text = next_tag.get_text(separator=' ', strip=True)
                if text:
                    results.append(text)
                next_tag = next_tag.find_next_sibling()
    return ' '.join(results)

full_animal_data = [["Animal Name", "Animal URL", "Description", "Habitat", "Diet", "Behavior", "Physical Traits", "Classification"]]

for index, href in enumerate(sampled):
    clean = href.replace('https://a-z-animals.com', '').strip('/')
    name  = clean.split('/')[-1].replace('-', ' ').title()
    url   = urljoin(BASE_URL, href)

    print(f"[{index + 1}/{len(sampled)}] Scraping: {name}")

    try:
        res = session.get(url, timeout=10)
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, 'html.parser')

        description = get_text_by_keywords(soup, ['overview', 'about', 'description', 'summary'])
        habitat     = get_text_by_keywords(soup, ['habitat', 'location', 'range', 'where'])
        diet        = get_text_by_keywords(soup, ['diet', 'food', 'eat', 'prey', 'feeding'])
        behavior    = get_text_by_keywords(soup, ['behavior', 'behaviour', 'lifestyle', 'activity', 'social'])
        physical    = get_text_by_keywords(soup, ['appearance', 'physical', 'size', 'weight', 'length', 'color'])
        classif     = get_text_by_keywords(soup, ['classification', 'taxonomy', 'scientific'])

        if not description:
            paras       = soup.find_all('p')
            description = ' '.join(p.get_text(strip=True) for p in paras[:3] if p.get_text(strip=True))

        full_animal_data.append([
            name, url,
            description or "No Data",
            habitat     or "No Data",
            diet        or "No Data",
            behavior    or "No Data",
            physical    or "No Data",
            classif     or "No Data",
        ])

    except Exception as e:
        print(f"Error: {e}")
        full_animal_data.append([name, url, "Error", "Error", "Error", "Error", "Error", "Error"])


# ========== Save to CSV ==========
with open(OUT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerows(full_animal_data)

print(f"\nDone! {len(full_animal_data)-1} animals → {OUT_FILE}")


# Start of PreprocessData.py
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

    # --- Web / UI artifacts ---
    'click', 'explore', 'map', 'show', 'read', 'learn', 'discover',
    'visit', 'page', 'article', 'website', 'link', 'photo', 'image',
    'copyright', 'reserved', 'right', 'source', 'reference', 'fact',

    # --- Filler adverbs & conjunctions ---
    'also', 'however', 'although', 'though', 'typically', 'generally',
    'commonly', 'usually', 'often', 'sometimes', 'occasionally',
    'therefore', 'thus', 'hence', 'furthermore', 'moreover',
    'additionally', 'nevertheless', 'nonetheless',

    # --- Generic verbs ---
    'make', 'made', 'use', 'used', 'using', 'able',
    'include', 'including', 'become', 'consider', 'get', 'give',
    'take', 'keep', 'let', 'put', 'seem', 'tell', 'try', 'mean',
    'call', 'called', 'known', 'refer',
    'animal', 'creature', 'organism', 'thing', 'example',
    'type', 'form', 'way', 'part', 'number', 'member',
    'world', 'nature', 'environment',
    'many', 'most', 'some', 'several', 'various', 'certain',
    'few', 'less', 'more', 'much', 'lot', 'enough',

    # --- Time references ---
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


# Start of AnimalSearch.py
import os
import re
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# =============================================================================
#  SECTION 1 — CONFIGURATION
# =============================================================================

PROJECT_DIR         = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_PATH      = os.path.normpath(os.path.join(PROJECT_DIR, '..', 'nltk_data'))
CSV_PATH            = os.path.join(PROJECT_DIR, 'csv', 'Animal_Data_Cleaned.csv')
TOP_N               = 5
RELEVANCE_THRESHOLD = 0.1

nltk.data.path.insert(0, NLTK_DATA_PATH)

# Columns to fill NaN and include in search text
CLEANED_COLS = [
    'Cleaned Description',
    'Cleaned Habitat',
    'Cleaned Diet',
    'Cleaned Behavior',
    'Cleaned Physical',
    'Cleaned Classification',
]

# =============================================================================
#  SECTION 2 — DATA LOADING
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {path}")

    for col in CLEANED_COLS:
        df[col] = df[col].fillna('')

    df['search'] = (
    (df['Cleaned Name'].astype(str) + ' ') * 5 +
    (df['Cleaned Physical'].astype(str) + ' ') * 2 +
    (df['Cleaned Behavior'].astype(str) + ' ') * 2 +
    (df['Cleaned Habitat'].astype(str) + ' ') * 2 +
    (df['Cleaned Diet'].astype(str) + ' ') * 2 +
    (df['Cleaned Classification'].astype(str) + ' ') * 2 +
    df['Cleaned Description'].astype(str)
)

    return df


# =============================================================================
#  SECTION 3 — INDEX BUILDING
# =============================================================================

def build_tfidf(df: pd.DataFrame):
    vectorizer   = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['search'])
    return vectorizer, tfidf_matrix


def build_inverted_index(df: pd.DataFrame, vectorizer, tfidf_matrix) -> dict:
    """Map each vocabulary term to its document frequency and postings list."""
    index = {}
    for col_idx, term in enumerate(vectorizer.get_feature_names_out()):
        doc_indices = tfidf_matrix[:, col_idx].nonzero()[0]
        index[term] = {
            'DF':       len(doc_indices),
        }
    return index


# =============================================================================
#  SECTION 4 — QUERY PROCESSING
# =============================================================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_query(query: str) -> str:
    tokens = re.sub(r'[^a-z\s]', ' ', query.lower()).split()
    return " ".join(lemmatizer.lemmatize(w) for w in tokens if w not in stop_words)

def expand_query_local(query, df, vectorizer, tfidf_matrix, top_k=3):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = scores.argsort()[-top_k:][::-1]

    feature_names = vectorizer.get_feature_names_out()
    new_terms = []

    for idx in top_idx:
        doc_vec = tfidf_matrix[idx].toarray().flatten()

        top_terms = doc_vec.argsort()[-2:][::-1]
        for t in top_terms:
            term = feature_names[t]
            if term not in query and term not in new_terms:
                new_terms.append(term)

    new_terms = new_terms[:5]

    expanded = query + " " + " ".join(new_terms)
    return expanded.strip()

# =============================================================================
#  SECTION 5 — SEARCH
# =============================================================================

def search(query: str, df: pd.DataFrame, vectorizer, tfidf_matrix,
           top_n: int = TOP_N) -> tuple:
    query_vec   = vectorizer.transform([query])
    scores      = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-top_n:][::-1]

    results = [
        {
            'Index': int(idx),
            'Name':  df.iloc[idx]['Animal Name'],
            'Score': float(scores[idx]),
            'URL':   df.iloc[idx]['Animal URL'],
        }
        for idx in top_indices
    ]
    return results, scores


# =============================================================================
#  SECTION 6 — EVALUATION METRICS
# =============================================================================

def precision_cal(retrieved: list, relevant: list, k: int) -> float:
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    return len(set(retrieved_k) & set(relevant)) / k


def recall_cal(retrieved: list, relevant: list, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def average_precision(retrieved: list, relevant: list) -> float:
    if not relevant:
        return 0.0
    score, hits = 0.0, 0.0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            hits  += 1
            score += hits / (i + 1)
    return score / len(relevant)


def get_relevant_docs(all_scores: np.ndarray,
                      threshold: float = RELEVANCE_THRESHOLD) -> list[int]:
    return [int(i) for i, s in enumerate(all_scores) if s >= threshold]


# =============================================================================
#  SECTION 7 — OUTPUT
# =============================================================================

DIVIDER      = "-" * 60
DIVIDER_BOLD = "=" * 60


def print_inverted_index(clean_q: str, inverted_index: dict) -> None:
    print(f"\n{DIVIDER}")
    print(f"  Inverted Index  |  query: '{clean_q}'")
    print(DIVIDER)
    print(f"  {'Term':<15} {'DF':<6}")
    print(DIVIDER)
    for term in clean_q.split():
        if term in inverted_index:
            entry = inverted_index[term]
            print(f"  {term:<15} {entry['DF']:<6}")
        else:
            print(f"  {term:<15} {'0':<6} (not found)")
    print(DIVIDER)


def print_results(results: list[dict], original_query: str, expanded_query: str = None) -> None:
    print(f"\n  {len(results)} results for '{original_query}'")
    if expanded_query and expanded_query != original_query:
        print(f"  Expanded query: '{expanded_query}'")
    print()
    for i, r in enumerate(results, start=1):
        print(f"Rank: {i} {r['Name']}  (similarity: {r['Score']:.4f})")
        print(f"        {r['URL']}")


def print_evaluation(results: list[dict], relevant_docs: list[int],
                     k: int) -> tuple[float, float, float]:
    retrieved = [r['Index'] for r in results]
    TP = len(set(retrieved[:k]) & set(relevant_docs))
    FP = k - TP
    FN = len(relevant_docs) - TP

    p_score  = precision_cal(retrieved, relevant_docs, k)
    r_score  = recall_cal(retrieved, relevant_docs, k)
    ap_score = average_precision(retrieved, relevant_docs)

    # --- per-rank detail ---
    print(f"\n{DIVIDER}")
    print(f"  Ranking Detail (Expanded Query)")
    print(DIVIDER)

    hits, sum_p = 0, 0.0
    matches, misses = [], []
    for i, idx in enumerate(retrieved):
        rank = i + 1
        if idx in relevant_docs:
            hits     += 1
            p_at_rank = hits / rank
            sum_p    += p_at_rank
            matches.append(f"  Rank {rank:>2}: Match    P{rank} = {hits}/{rank} = {p_at_rank:.2f}")
        else:
            misses.append(f"  Rank {rank:>2}: No match")

    for line in sorted(matches + misses, key=lambda l: int(re.search(r'\d+', l).group())):
        print(line)

    # --- AP calculation ---
    total_relevant = len(relevant_docs)
    print(DIVIDER)
    if total_relevant > 0:
        print(f"  Sum of Precisions : {sum_p:.2f}")
        print(f"  Total Relevant    : {total_relevant} documents")
        print(f"  AP = {sum_p:.2f} / {total_relevant} = {ap_score:.2f}")
    else:
        print("  AP = 0.00  (no relevant documents found)")

    # --- TP / FP / FN ---
    print(f"  TP : {TP}   FP : {FP}   FN : {FN}")

    # --- Precision & Recall calculation process ---
    print(f"\n{DIVIDER}")
    print(f"  Precision & Recall Calculation")
    print(DIVIDER)
    print(f"  Precision = TP/(TP+FP) = {TP}/({TP}+{FP}) = {TP}/{TP+FP} = {p_score:.2f}")
    print(f"  Recall    = TP/(TP+FN) = {TP}/({TP}+{FN}) = {TP}/{TP+FN} = {r_score:.2f}")
    print(DIVIDER)

    # --- summary bar ---
    print(f"\n{DIVIDER_BOLD}")
    print(f"  Precision: {p_score:.2f}   Recall: {r_score:.2f}   AP: {ap_score:.2f}")
    print(DIVIDER_BOLD)

    return p_score, r_score, ap_score


def print_session_history(history: list[dict]) -> None:
    print(f"\n{DIVIDER}")
    print(f"  History")
    print(DIVIDER)
    print(f"  {'Query':<22} {'Precision':>10} {'Recall':>8} {'AP':>6}")
    print(DIVIDER)
    ap_list = []
    for h in history:
        print(f"  {h['query']:<22} {h['p']:>10.2f} {h['r']:>8.2f} {h['ap']:>6.2f}")
        ap_list.append(h['ap'])
    print(DIVIDER)
    print(f"  MAP (Mean Average Precision) : {np.mean(ap_list):.2f}")
    print(DIVIDER)


# =============================================================================
#  SECTION 8 — MAIN LOOP
# =============================================================================

def main() -> None:
    # --- startup: load data and build indices ---
    df                          = load_data(CSV_PATH)
    vectorizer, tfidf_matrix    = build_tfidf(df)
    inverted_index              = build_inverted_index(df, vectorizer, tfidf_matrix)
    session_history: list[dict] = []

    while True:
        print(f"\n{'Animal Finder':^60}")
        print(DIVIDER)

        user_query = input("\n  Search: ").strip()

        if not user_query:
            print("Please enter a search term.")
            continue

        # --- query processing ---
        clean_q = clean_query(user_query)
        if not clean_q:
            print("Please enter a valid animal trait or habitat.")
            continue

        if clean_q != user_query.lower():
            print(f"\nCleaned  : '{clean_q}'")

        expanded_q = expand_query_local(clean_q, df, vectorizer, tfidf_matrix)
        if expanded_q != clean_q:
            print(f"Expanded : '{expanded_q}'")

        # --- inverted index display ---
        print_inverted_index(expanded_q, inverted_index)

        # --- search ---
        original_results, _ = search(clean_q, df, vectorizer, tfidf_matrix, top_n=TOP_N)
        results, all_scores = search(expanded_q, df, vectorizer, tfidf_matrix, top_n=TOP_N)

        if not results:
            print(f"\nNo results found for '{user_query}'")
            print(DIVIDER)
            continue

        # --- results display ---
        print(f"\n  [Original Query]")
        print_results(original_results, user_query)

        print(f"\n  [Expanded Query]")
        print_results(results, user_query, expanded_q)

        # --- evaluation ---
        relevant_docs        = get_relevant_docs(all_scores)
        p_score, r_score, ap = print_evaluation(results, relevant_docs, k=TOP_N)

        # --- session history ---
        session_history.append({'query': user_query, 'p': p_score, 'r': r_score, 'ap': ap})
        print_session_history(session_history)


if __name__ == '__main__':
    main()