# =============================================================================
#  Animal Finder — Information Retrieval System
#  Uses TF-IDF + Cosine Similarity to search animals by trait, habitat, or diet
# =============================================================================

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
#  All constants and paths are defined here.
#  Adjust TOP_N or RELEVANCE_THRESHOLD to tune search behaviour.
# =============================================================================

PROJECT_DIR         = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_PATH      = os.path.normpath(os.path.join(PROJECT_DIR, '..', 'nltk_data'))
CSV_PATH            = os.path.join(PROJECT_DIR, 'csv', 'Animal_Data_Cleaned.csv')
TOP_N               = 10       # Number of results to return per query
RELEVANCE_THRESHOLD = 0.05     # Minimum similarity score to count as relevant

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

# Synonym map for query expansion — each key expands to related terms at search time
SYNONYMS = {
    'jungle':    ['rainforest', 'tropical', 'forest', 'tree'],
    'forest':    ['woodland', 'jungle', 'rainforest', 'tree'],
    'ocean':     ['sea', 'marine', 'aquatic', 'water', 'deep'],
    'sea':       ['ocean', 'marine', 'aquatic', 'coral', 'reef'],
    'desert':    ['arid', 'dry', 'sand', 'savanna'],
    'savanna':   ['grassland', 'plain', 'africa', 'dry'],
    'grassland': ['plain', 'savanna', 'prairie', 'meadow'],
    'arctic':    ['polar', 'cold', 'ice', 'snow', 'tundra'],
    'mountain':  ['highland', 'cliff', 'rocky', 'alpine'],
    'river':     ['freshwater', 'stream', 'lake', 'wetland'],
    'australia': ['marsupial', 'outback', 'eucalyptus'],
    'meat':      ['carnivore', 'prey', 'hunt', 'flesh'],
    'plant':     ['herbivore', 'leaf', 'grass', 'vegetation', 'foliage'],
    'fish':      ['aquatic', 'salmon', 'tuna', 'fishing'],
    'insect':    ['bug', 'ant', 'beetle', 'invertebrate'],
    'carnivore': ['predator', 'hunt', 'meat', 'prey'],
    'herbivore': ['plant', 'grass', 'leaf', 'vegetation'],
    'omnivore':  ['plant', 'meat', 'fruit', 'insect'],
    'big':       ['large', 'giant', 'huge', 'massive', 'heavy'],
    'small':     ['tiny', 'little', 'miniature', 'dwarf'],
    'fast':      ['speed', 'quick', 'agile', 'swift', 'sprint'],
    'slow':      ['sluggish', 'lazy', 'lethargic'],
    'fly':       ['wing', 'bird', 'aerial', 'flight', 'soar'],
    'swim':      ['aquatic', 'fin', 'water', 'marine', 'dive'],
    'jump':      ['leap', 'hop', 'spring', 'bounce', 'kangaroo'],
    'climb':     ['tree', 'arboreal', 'claw', 'grip'],
    'nocturnal': ['night', 'dark', 'owl', 'bat'],
    'stripe':    ['striped', 'zebra', 'tiger', 'pattern'],
    'spot':      ['spotted', 'leopard', 'cheetah', 'dot'],
    'mane':      ['lion', 'horse', 'hair', 'neck'],
    'tusk':      ['elephant', 'ivory', 'walrus'],
    'horn':      ['rhino', 'antler', 'deer', 'goat'],
    'shell':     ['turtle', 'tortoise', 'snail', 'crab'],
    'poison':    ['venom', 'toxic', 'snake', 'frog'],
    'venom':     ['poison', 'toxic', 'bite', 'sting'],
    'social':    ['pack', 'herd', 'group', 'colony', 'flock'],
    'solitary':  ['alone', 'lone', 'independent'],
    'hunt':      ['predator', 'prey', 'chase', 'carnivore'],
    'migrate':   ['seasonal', 'travel', 'journey', 'flock'],
    'mammal':    ['warm', 'fur', 'hair', 'milk', 'breast'],
    'reptile':   ['scale', 'cold', 'lizard', 'snake', 'crocodile'],
    'bird':      ['feather', 'wing', 'beak', 'fly', 'egg'],
    'amphibian': ['frog', 'toad', 'salamander', 'water', 'land'],
}


# =============================================================================
#  SECTION 2 — DATA LOADING
#  Reads the CSV file and builds a weighted Search_Text column.
#  Name and physical/behavioral fields are weighted higher for better relevance.
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load the animal CSV and construct a weighted Search_Text column."""
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {path}")

    for col in CLEANED_COLS:
        df[col] = df[col].fillna('')

    # Weight: Name x5, Physical x2, Behavior x2, rest x1
    df['Search_Text'] = (
        (df['Cleaned Name'].astype(str) + ' ') * 5 +
        (df['Cleaned Physical'].astype(str) + ' ') * 2 +
        (df['Cleaned Behavior'].astype(str) + ' ') * 2 +
        df['Cleaned Description'].astype(str) + ' ' +
        df['Cleaned Habitat'].astype(str) + ' ' +
        df['Cleaned Diet'].astype(str) + ' ' +
        df['Cleaned Classification'].astype(str)
    )

    return df


# =============================================================================
#  SECTION 3 — INDEX BUILDING
#  Builds two structures used at search time:
#    - TF-IDF matrix : numerical representation of each document
#    - Inverted index : maps each term to documents that contain it (for display)
# =============================================================================

def build_tfidf(df: pd.DataFrame):
    """Fit TF-IDF on Search_Text. Returns (vectorizer, tfidf_matrix)."""
    vectorizer   = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Search_Text'])
    return vectorizer, tfidf_matrix


def build_inverted_index(df: pd.DataFrame, vectorizer, tfidf_matrix) -> dict:
    """Map each vocabulary term to its document frequency and postings list."""
    index = {}
    for col_idx, term in enumerate(vectorizer.get_feature_names_out()):
        doc_indices = tfidf_matrix[:, col_idx].nonzero()[0]
        index[term] = {
            'DF':       len(doc_indices),
            'Postings': [df.iloc[i]['Animal Name'] for i in doc_indices],
        }
    return index


# =============================================================================
#  SECTION 4 — QUERY PROCESSING
#  Prepares user input before it is passed to the search engine:
#    1. clean_query  : lowercase, remove noise, lemmatize, strip stopwords
#    2. expand_query : append synonyms to broaden recall
# =============================================================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_query(query: str) -> str:
    """Normalize query: lowercase, strip non-alpha, lemmatize, remove stopwords."""
    tokens = re.sub(r'[^a-z\s]', ' ', query.lower()).split()
    return " ".join(lemmatizer.lemmatize(w) for w in tokens if w not in stop_words)


def expand_query(query: str) -> str:
    """Expand each token with synonyms from the SYNONYMS dictionary."""
    tokens = []
    for word in query.lower().split():
        tokens.append(word)
        tokens.extend(SYNONYMS.get(word, []))
    return " ".join(tokens)


# =============================================================================
#  SECTION 5 — SEARCH
#  Transforms the processed query into a TF-IDF vector and ranks all documents
#  by cosine similarity. Returns the top_n results and the full score array.
# =============================================================================

def search(query: str, df: pd.DataFrame, vectorizer, tfidf_matrix,
           top_n: int = TOP_N) -> tuple:
    """Rank documents by cosine similarity and return top results + all scores."""
    expanded    = expand_query(query)
    query_vec   = vectorizer.transform([expanded])
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
#  Measures retrieval quality at rank k using standard IR metrics:
#    - Precision@K  : fraction of retrieved docs that are relevant
#    - Recall@K     : fraction of all relevant docs that were retrieved
#    - Average Precision (AP) : area under the precision-recall curve
#    - MAP          : mean AP across all queries in the session
#
#  Relevant documents are defined by similarity score >= RELEVANCE_THRESHOLD
#  (rather than regex matching, which tends to overcount).
# =============================================================================

def precision_at_k(retrieved: list, relevant: list, k: int) -> float:
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    return len(set(retrieved_k) & set(relevant)) / k


def recall_at_k(retrieved: list, relevant: list, k: int) -> float:
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
    """Return indices of documents whose similarity score meets the threshold."""
    return [int(i) for i, s in enumerate(all_scores) if s >= threshold]


# =============================================================================
#  SECTION 7 — DISPLAY
#  All terminal output lives here. Kept separate so logic sections above
#  remain free of print statements and are easier to test or reuse.
# =============================================================================

DIVIDER      = "-" * 60
DIVIDER_BOLD = "=" * 60


def print_inverted_index(clean_q: str, inverted_index: dict) -> None:
    """Show which documents contain each query term and how many."""
    print(f"\n{DIVIDER}")
    print(f"  Inverted Index  |  query: '{clean_q}'")
    print(DIVIDER)
    print(f"  {'Term':<15} {'DF':<6} Postings (first 10)")
    print(DIVIDER)
    for term in clean_q.split():
        if term in inverted_index:
            entry        = inverted_index[term]
            postings_str = ", ".join(entry['Postings'][:10])
            if entry['DF'] > 10:
                postings_str += " ..."
            print(f"  {term:<15} {entry['DF']:<6} {postings_str}")
        else:
            print(f"  {term:<15} {'0':<6} (not found)")
    print(DIVIDER)


def print_results(results: list[dict], original_query: str) -> None:
    """Display ranked search results with name, similarity score, and URL."""
    print(f"\n  {len(results)} results for '{original_query}'\n")
    for i, r in enumerate(results, start=1):
        print(f"  [{i:>2}] {r['Name']}  (similarity: {r['Score']:.4f})")
        print(f"        {r['URL']}")


def print_evaluation(results: list[dict], relevant_docs: list[int],
                     k: int) -> tuple[float, float, float]:
    """Print per-rank match detail and summary metrics (Precision, Recall, AP)."""
    retrieved = [r['Index'] for r in results]
    TP = len(set(retrieved[:k]) & set(relevant_docs))
    FP = k - TP
    FN = len(relevant_docs) - TP

    p_score  = precision_at_k(retrieved, relevant_docs, k)
    r_score  = recall_at_k(retrieved, relevant_docs, k)
    ap_score = average_precision(retrieved, relevant_docs)

    # --- per-rank detail ---
    print(f"\n{DIVIDER}")
    print(f"  Rank-by-rank breakdown")
    print(DIVIDER)

    hits, sum_p = 0, 0.0
    matches, misses = [], []
    for i, idx in enumerate(retrieved):
        rank = i + 1
        if idx in relevant_docs:
            hits     += 1
            p_at_rank = hits / rank
            sum_p    += p_at_rank
            matches.append(f"  Rank {rank:>2}: Match    P@{rank} = {hits}/{rank} = {p_at_rank:.2f}")
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
    print(f"\n  TP : {TP}   FP : {FP}   FN : {FN}")

    # --- summary bar ---
    print(f"\n{DIVIDER_BOLD}")
    print(f"  Precision@{k}: {p_score:.2f}   Recall@{k}: {r_score:.2f}   AP: {ap_score:.2f}")
    print(DIVIDER_BOLD)

    return p_score, r_score, ap_score


def print_session_history(history: list[dict]) -> None:
    """Show all queries this session and the running MAP score."""
    print(f"\n{DIVIDER}")
    print(f"  Session History")
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
#  Entry point. Initializes all components once, then runs the interactive
#  search loop until the user types 'q' or 'exit'.
# =============================================================================

def main() -> None:
    # --- startup: load data and build indices ---
    df                          = load_data(CSV_PATH)
    vectorizer, tfidf_matrix    = build_tfidf(df)
    inverted_index              = build_inverted_index(df, vectorizer, tfidf_matrix)
    session_history: list[dict] = []

    while True:
        print(f"\n{'  Animal Finder':^60}")
        print(f"{'  type q or exit to quit':^60}")
        print(DIVIDER)

        user_query = input("\n  Search: ").strip()

        if user_query.lower() in {'q', 'exit', 'quit'}:
            print("\n  Goodbye.\n")
            break

        if not user_query:
            print("  Please enter a search term.")
            continue

        # --- query processing ---
        clean_q = clean_query(user_query)
        if not clean_q:
            print("  Please enter a valid animal trait or habitat.")
            continue

        if clean_q != user_query.lower():
            print(f"\n  Cleaned  : '{clean_q}'")

        expanded_q = expand_query(clean_q)
        if expanded_q != clean_q:
            print(f"  Expanded : '{expanded_q}'")

        # --- inverted index display ---
        print_inverted_index(clean_q, inverted_index)

        # --- search ---
        results, all_scores = search(clean_q, df, vectorizer, tfidf_matrix, top_n=TOP_N)

        if not results:
            print(f"\n  No results found for '{user_query}'")
            print(DIVIDER)
            continue

        # --- results display ---
        print_results(results, user_query)

        # --- evaluation ---
        relevant_docs        = get_relevant_docs(all_scores)
        p_score, r_score, ap = print_evaluation(results, relevant_docs, k=TOP_N)

        # --- session history ---
        session_history.append({'query': user_query, 'p': p_score, 'r': r_score, 'ap': ap})
        print_session_history(session_history)


if __name__ == '__main__':
    main()