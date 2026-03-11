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