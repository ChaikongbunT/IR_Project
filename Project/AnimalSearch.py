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

# Synonym map for query expansion — each key expands to related terms at search time
SYNONYMS = {

    # ------------------------------------------------------------------
    # HABITAT — where the animal lives
    # ------------------------------------------------------------------
    'jungle':     ['rainforest', 'tropical', 'forest', 'canopy', 'tree', 'dense'],
    'forest':     ['woodland', 'jungle', 'rainforest', 'tree', 'canopy', 'boreal'],
    'rainforest': ['jungle', 'tropical', 'humid', 'canopy', 'dense', 'forest'],
    'woodland':   ['forest', 'tree', 'shrub', 'temperate'],
    'ocean':      ['sea', 'marine', 'aquatic', 'deep', 'pelagic', 'offshore'],
    'sea':        ['ocean', 'marine', 'aquatic', 'coral', 'reef', 'coastal'],
    'marine':     ['ocean', 'sea', 'aquatic', 'saltwater', 'coastal', 'reef'],
    'coral':      ['reef', 'tropical', 'marine', 'sea', 'ocean'],
    'reef':       ['coral', 'marine', 'tropical', 'shallow', 'sea'],
    'river':      ['freshwater', 'stream', 'creek', 'lake', 'wetland', 'riparian'],
    'lake':       ['freshwater', 'river', 'pond', 'wetland', 'inland'],
    'wetland':    ['swamp', 'marsh', 'river', 'freshwater', 'bog', 'mangrove'],
    'swamp':      ['wetland', 'marsh', 'mangrove', 'muddy', 'freshwater'],
    'mangrove':   ['coastal', 'wetland', 'swamp', 'tropical', 'estuary'],
    'desert':     ['arid', 'dry', 'sand', 'dune', 'barren', 'scrub'],
    'arid':       ['desert', 'dry', 'sand', 'scrub', 'sparse'],
    'savanna':    ['grassland', 'plain', 'africa', 'dry', 'shrub', 'open'],
    'grassland':  ['plain', 'savanna', 'prairie', 'meadow', 'steppe', 'open'],
    'prairie':    ['grassland', 'plain', 'meadow', 'open', 'north america'],
    'tundra':     ['arctic', 'polar', 'cold', 'barren', 'permafrost', 'alpine'],
    'arctic':     ['polar', 'cold', 'ice', 'snow', 'tundra', 'frozen'],
    'polar':      ['arctic', 'antarctic', 'ice', 'cold', 'snow', 'tundra'],
    'mountain':   ['highland', 'cliff', 'rocky', 'alpine', 'peak', 'slope'],
    'highland':   ['mountain', 'alpine', 'rocky', 'plateau', 'elevation'],
    'alpine':     ['mountain', 'highland', 'cold', 'rocky', 'snow'],
    'cave':       ['underground', 'dark', 'bat', 'subterranean', 'rocky'],
    'underground':['burrow', 'cave', 'tunnel', 'subterranean', 'soil'],
    'burrow':     ['underground', 'tunnel', 'soil', 'dig', 'mole'],
    'coastal':    ['shore', 'beach', 'sea', 'marine', 'tidal', 'cliff'],
    'island':     ['tropical', 'isolated', 'coastal', 'endemic'],
    'tree':       ['arboreal', 'forest', 'canopy', 'branch', 'climb'],

    # ------------------------------------------------------------------
    # GEOGRAPHY — regions and continents
    # ------------------------------------------------------------------
    'africa':       ['savanna', 'grassland', 'safari', 'tropical', 'sub-saharan'],
    'asia':         ['tropical', 'india', 'china', 'southeast', 'himalaya'],
    'india':        ['asia', 'tropical', 'bengal', 'himalaya', 'monsoon'],
    'china':        ['asia', 'panda', 'bamboo', 'himalaya'],
    'australia':    ['marsupial', 'outback', 'eucalyptus', 'oceania', 'reef'],
    'europe':       ['temperate', 'woodland', 'boreal', 'alpine'],
    'amazon':       ['rainforest', 'tropical', 'south america', 'jungle', 'river'],
    'antarctic':    ['polar', 'cold', 'ice', 'penguin', 'snow', 'frozen'],
    'arctic':       ['polar', 'cold', 'ice', 'snow', 'tundra', 'frozen'],
    'southeast':    ['tropical', 'asia', 'jungle', 'rainforest', 'humid'],
    'north america':['prairie', 'forest', 'temperate', 'mountain', 'river'],
    'south america':['amazon', 'rainforest', 'tropical', 'andes', 'jungle'],

    # ------------------------------------------------------------------
    # DIET — what the animal eats
    # ------------------------------------------------------------------
    'meat':      ['carnivore', 'prey', 'hunt', 'flesh', 'kill', 'predator'],
    'plant':     ['herbivore', 'leaf', 'grass', 'vegetation', 'foliage', 'browse'],
    'grass':     ['herbivore', 'graze', 'meadow', 'vegetation', 'plant'],
    'fruit':     ['omnivore', 'berry', 'seed', 'tropical', 'plant'],
    'seed':      ['bird', 'rodent', 'granivore', 'grain', 'plant'],
    'fish':      ['piscivore', 'aquatic', 'marine', 'salmon', 'water', 'fishing'],
    'insect':    ['invertebrate', 'bug', 'ant', 'beetle', 'larvae', 'arthropod'],
    'bug':       ['insect', 'invertebrate', 'beetle', 'larvae', 'spider'],
    'worm':      ['invertebrate', 'soil', 'underground', 'earthworm'],
    'nectar':    ['flower', 'pollinator', 'bee', 'butterfly', 'hummingbird'],
    'blood':     ['parasite', 'vampire', 'bat', 'leech', 'mosquito'],
    'carrion':   ['scavenger', 'dead', 'vulture', 'hyena', 'decay'],
    'plankton':  ['filter', 'marine', 'whale', 'aquatic', 'microscopic'],
    'bamboo':    ['panda', 'plant', 'herbivore', 'asia', 'china'],
    'carnivore': ['predator', 'hunt', 'meat', 'prey', 'kill', 'flesh'],
    'herbivore': ['plant', 'grass', 'leaf', 'vegetation', 'graze', 'browse'],
    'omnivore':  ['plant', 'meat', 'fruit', 'insect', 'flexible'],
    'scavenger': ['carrion', 'dead', 'vulture', 'hyena', 'opportunistic'],
    'predator':  ['hunt', 'carnivore', 'apex', 'prey', 'kill', 'chase'],
    'prey':      ['hunted', 'victim', 'small', 'flee', 'escape'],
    'filter':    ['plankton', 'whale', 'flamingo', 'baleen', 'marine'],
    'graze':     ['grass', 'herbivore', 'cattle', 'plain', 'meadow'],

    # ------------------------------------------------------------------
    # BEHAVIOR — how the animal acts
    # ------------------------------------------------------------------
    'hunt':      ['predator', 'chase', 'carnivore', 'ambush', 'stalk', 'prey'],
    'stalk':     ['hunt', 'predator', 'ambush', 'stealth', 'cat', 'sneak'],
    'ambush':    ['stalk', 'hunt', 'camouflage', 'wait', 'predator'],
    'nocturnal': ['night', 'dark', 'owl', 'bat', 'crepuscular', 'darkness'],
    'diurnal':   ['day', 'daytime', 'active', 'sun', 'light'],
    'migrate':   ['seasonal', 'travel', 'journey', 'flock', 'long distance'],
    'hibernate': ['winter', 'sleep', 'dormant', 'cold', 'burrow'],
    'social':    ['pack', 'herd', 'colony', 'flock', 'group', 'pod', 'troop'],
    'solitary':  ['alone', 'lone', 'independent', 'territorial', 'isolated'],
    'territorial':['solitary', 'defend', 'mark', 'aggressive', 'dominant'],
    'pack':      ['wolf', 'social', 'group', 'hunt', 'cooperative'],
    'herd':      ['social', 'group', 'cattle', 'elephant', 'plain'],
    'flock':     ['bird', 'social', 'group', 'migrate', 'sky'],
    'colony':    ['social', 'ant', 'bee', 'bat', 'group', 'cooperative'],
    'camouflage':['hide', 'blend', 'color', 'pattern', 'ambush', 'stealth'],
    'venomous':  ['poison', 'toxic', 'bite', 'sting', 'snake', 'dangerous'],
    'poison':    ['venom', 'toxic', 'frog', 'dangerous', 'lethal'],
    'venom':     ['poison', 'toxic', 'bite', 'sting', 'fang', 'gland'],
    'aggressive': ['territorial', 'dominant', 'attack', 'dangerous', 'fierce'],
    'defensive': ['camouflage', 'shell', 'spine', 'armor', 'flee'],
    'dig':       ['burrow', 'underground', 'soil', 'mole', 'tunnel'],
    'swim':      ['aquatic', 'fin', 'water', 'marine', 'dive', 'streamlined'],
    'dive':      ['swim', 'deep', 'aquatic', 'bird', 'marine', 'plunge'],
    'fly':       ['wing', 'aerial', 'flight', 'soar', 'glide', 'bird'],
    'glide':     ['fly', 'aerial', 'wing', 'soar', 'membrane'],
    'climb':     ['arboreal', 'tree', 'claw', 'grip', 'monkey', 'gecko'],
    'jump':      ['leap', 'hop', 'spring', 'bounce', 'kangaroo', 'frog'],
    'run':       ['fast', 'sprint', 'chase', 'speed', 'gallop', 'flee'],
    'fast':      ['speed', 'quick', 'agile', 'swift', 'sprint', 'cheetah'],
    'slow':      ['sluggish', 'lethargic', 'turtle', 'sloth', 'crawl'],

    # ------------------------------------------------------------------
    # PHYSICAL TRAITS — appearance and body features
    # ------------------------------------------------------------------
    'big':       ['large', 'giant', 'huge', 'massive', 'heavy', 'bulky'],
    'large':     ['big', 'giant', 'massive', 'heavy', 'bulky', 'dominant'],
    'giant':     ['large', 'massive', 'biggest', 'dominant', 'enormous'],
    'small':     ['tiny', 'miniature', 'dwarf', 'little', 'lightweight'],
    'tiny':      ['small', 'miniature', 'dwarf', 'micro', 'lightweight'],
    'long':      ['elongated', 'slender', 'stretch', 'neck', 'snake'],
    'tall':      ['long', 'giraffe', 'large', 'height', 'neck'],
    'heavy':     ['large', 'massive', 'fat', 'bulky', 'elephant', 'hippo'],
    'wing':      ['fly', 'bird', 'aerial', 'feather', 'bat', 'membrane'],
    'feather':   ['bird', 'wing', 'plume', 'colorful', 'flight'],
    'fur':       ['mammal', 'warm', 'coat', 'thick', 'fluffy', 'hair'],
    'scale':     ['reptile', 'fish', 'armor', 'skin', 'cold'],
    'shell':     ['turtle', 'tortoise', 'snail', 'crab', 'armor', 'hard'],
    'spine':     ['hedgehog', 'porcupine', 'defensive', 'sharp', 'armor'],
    'horn':      ['rhino', 'antler', 'deer', 'goat', 'bovine', 'keratin'],
    'antler':    ['deer', 'elk', 'moose', 'horn', 'seasonal', 'male'],
    'tusk':      ['elephant', 'ivory', 'walrus', 'boar', 'elongated'],
    'fang':      ['venom', 'snake', 'predator', 'bite', 'sharp'],
    'claw':      ['predator', 'sharp', 'grip', 'climb', 'talon', 'cat'],
    'talon':     ['bird', 'raptor', 'claw', 'sharp', 'eagle', 'owl'],
    'beak':      ['bird', 'bill', 'feed', 'sharp', 'peck'],
    'bill':      ['beak', 'bird', 'duck', 'platypus', 'flat'],
    'trunk':     ['elephant', 'nose', 'long', 'flexible', 'mammal'],
    'tail':      ['balance', 'monkey', 'cat', 'long', 'appendage'],
    'mane':      ['lion', 'horse', 'male', 'hair', 'neck', 'pride'],
    'pouch':     ['marsupial', 'kangaroo', 'koala', 'australia', 'joey'],
    'fin':       ['fish', 'aquatic', 'swim', 'marine', 'dolphin', 'shark'],
    'gill':      ['fish', 'aquatic', 'breathe', 'water', 'amphibian'],
    'stripe':    ['striped', 'zebra', 'tiger', 'pattern', 'linear'],
    'spot':      ['spotted', 'leopard', 'cheetah', 'dot', 'pattern'],
    'color':     ['colorful', 'bright', 'camouflage', 'pattern', 'pigment'],
    'colorful':  ['bright', 'vibrant', 'tropical', 'bird', 'frog', 'reef'],
    'bright':    ['colorful', 'vibrant', 'warning', 'tropical', 'bird'],

    # ------------------------------------------------------------------
    # COLOR
    # ------------------------------------------------------------------
    'black':   ['dark', 'melanistic', 'panther', 'shadow', 'night'],
    'white':   ['pale', 'albino', 'snow', 'arctic', 'light', 'polar'],
    'red':     ['bright', 'warning', 'colorful', 'tropical', 'vibrant'],
    'orange':  ['bright', 'colorful', 'tropical', 'warning', 'tiger'],
    'yellow':  ['bright', 'colorful', 'tropical', 'warning', 'bee'],
    'green':   ['camouflage', 'tropical', 'forest', 'frog', 'lizard'],
    'blue':    ['colorful', 'marine', 'tropical', 'bird', 'reef'],
    'brown':   ['camouflage', 'earth', 'woodland', 'bear', 'deer'],
    'grey':    ['elephant', 'wolf', 'camouflage', 'stone', 'dolphin'],
    'gray':    ['elephant', 'wolf', 'camouflage', 'stone', 'dolphin'],

    # ------------------------------------------------------------------
    # CLASSIFICATION — taxonomy
    # ------------------------------------------------------------------
    'mammal':     ['warm', 'fur', 'hair', 'milk', 'vertebrate', 'placental'],
    'reptile':    ['scale', 'cold', 'lizard', 'snake', 'crocodile', 'ectotherm'],
    'bird':       ['feather', 'wing', 'beak', 'fly', 'egg', 'avian'],
    'amphibian':  ['frog', 'toad', 'salamander', 'water', 'land', 'moist'],
    'fish':       ['aquatic', 'gill', 'fin', 'scale', 'freshwater', 'marine'],
    'insect':     ['invertebrate', 'bug', 'arthropod', 'six', 'exoskeleton'],
    'invertebrate':['insect', 'spider', 'crab', 'worm', 'no backbone'],
    'marsupial':  ['pouch', 'australia', 'kangaroo', 'koala', 'joey'],
    'primate':    ['monkey', 'ape', 'human', 'social', 'intelligent'],
    'rodent':     ['mouse', 'rat', 'squirrel', 'small', 'gnaw'],
    'feline':     ['cat', 'lion', 'tiger', 'leopard', 'jaguar', 'carnivore'],
    'canine':     ['dog', 'wolf', 'fox', 'pack', 'social'],
    'raptor':     ['bird', 'eagle', 'hawk', 'owl', 'talon', 'predator'],
    'shark':      ['marine', 'predator', 'fish', 'ocean', 'apex'],
    'whale':      ['marine', 'mammal', 'large', 'ocean', 'filter', 'cetacean'],
    'dolphin':    ['marine', 'mammal', 'social', 'intelligent', 'cetacean'],
    'snake':      ['reptile', 'scale', 'venom', 'elongated', 'slither'],
    'lizard':     ['reptile', 'scale', 'cold', 'climb', 'desert'],
    'frog':       ['amphibian', 'jump', 'water', 'poison', 'moist'],
    'spider':     ['invertebrate', 'arachnid', 'web', 'eight', 'venom'],
    'crab':       ['invertebrate', 'marine', 'shell', 'claw', 'coastal'],
    'butterfly':  ['insect', 'wing', 'colorful', 'flower', 'migrate', 'nectar'],
    'bee':        ['insect', 'colony', 'nectar', 'flower', 'social', 'sting'],

    # ------------------------------------------------------------------
    # CONSERVATION STATUS
    # ------------------------------------------------------------------
    'endangered': ['threatened', 'rare', 'extinction', 'protected', 'vulnerable'],
    'rare':       ['endangered', 'uncommon', 'scarce', 'threatened', 'few'],
    'extinct':    ['gone', 'lost', 'prehistoric', 'fossil', 'disappeared'],
    'protected':  ['endangered', 'conservation', 'reserve', 'sanctuary'],
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

    df['Search_Text'] = (
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
#    - Precision-K  : fraction of retrieved docs that are relevant
#    - Recall-K     : fraction of all relevant docs that were retrieved
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
    print(f"  {'Term':<15} {'DF':<6}")
    print(DIVIDER)
    for term in clean_q.split():
        if term in inverted_index:
            entry = inverted_index[term]
            print(f"  {term:<15} {entry['DF']:<6}")
        else:
            print(f"  {term:<15} {'0':<6} (not found)")
    print(DIVIDER)


def print_results(results: list[dict], original_query: str) -> None:
    """Display ranked search results with name, similarity score, and URL."""
    print(f"\n  {len(results)} results for '{original_query}'\n")
    for i, r in enumerate(results, start=1):
        print(f"[{i:>2}] {r['Name']}  (similarity: {r['Score']:.4f})")
        print(f"     {r['URL']}")


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
    print(f"  Ranking Detail")
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
    print(f"\n  TP : {TP}   FP : {FP}   FN : {FN}")

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
    """Show all queries this session and the running MAP score."""
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

        expanded_q = expand_query(clean_q)
        if expanded_q != clean_q:
            print(f"Expanded : '{expanded_q}'")

        # --- inverted index display ---
        print_inverted_index(clean_q, inverted_index)

        # --- search ---
        results, all_scores = search(clean_q, df, vectorizer, tfidf_matrix, top_n=TOP_N)

        if not results:
            print(f"\nNo results found for '{user_query}'")
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