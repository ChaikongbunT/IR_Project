import os
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ========== PATH CONFIG ==========
PROJECT_DIR    = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_PATH = os.path.normpath(os.path.join(PROJECT_DIR, '..', 'nltk_data'))
CSV_DIR        = os.path.join(PROJECT_DIR, 'csv')
IN_FILE        = os.path.join(CSV_DIR, 'Animal_Data_Cleaned.csv')
nltk.data.path.insert(0, NLTK_DATA_PATH)
# ==================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ==================== Evaluation Metrics ====================
def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    if not retrieved_k: return 0.0
    return len(set(retrieved_k).intersection(set(relevant))) / k

def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    if not relevant: return 0.0
    return len(set(retrieved_k).intersection(set(relevant))) / len(relevant)

def average_precision(retrieved, relevant):
    if not relevant: return 0.0
    score, hits = 0.0, 0.0
    for i, p in enumerate(retrieved):
        if p in relevant:
            hits  += 1.0
            score += hits / (i + 1.0)
    return score / len(relevant)

# ==================== Load Data ====================
try:
    df = pd.read_csv(IN_FILE, encoding='utf-8-sig')
    for col in ['Cleaned Description', 'Cleaned Habitat', 'Cleaned Diet',
                'Cleaned Behavior', 'Cleaned Physical', 'Cleaned Classification']:
        df[col] = df[col].fillna('')
except FileNotFoundError:
    print(f"{IN_FILE} not found.")
    exit()

# ==================== Build TF-IDF ====================
df['Search_Text'] = (
    (df['Cleaned Name'].astype(str) + ' ') * 5 +
    (df['Cleaned Physical'].astype(str) + ' ') * 2 +
    (df['Cleaned Behavior'].astype(str) + ' ') * 2 +
    df['Cleaned Description'].astype(str) + ' ' +
    df['Cleaned Habitat'].astype(str) + ' ' +
    df['Cleaned Diet'].astype(str) + ' ' +
    df['Cleaned Classification'].astype(str)
)

vectorizer   = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Search_Text'])

# ==================== Build Inverted Index ====================
feature_names = vectorizer.get_feature_names_out()
inverted_index = {}
for col_idx, term in enumerate(feature_names):
    doc_indices = tfidf_matrix[:, col_idx].nonzero()[0]
    inverted_index[term] = {
        'DF':       len(doc_indices),
        'Postings': [df.iloc[i]['Animal Name'] for i in doc_indices]
    }

# ==================== Synonyms for Query Expansion ====================
synonyms_dict = {
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

def expand_query(query):
    expanded = []
    for word in query.lower().split():
        expanded.append(word)
        if word in synonyms_dict:
            expanded.extend(synonyms_dict[word])
    return " ".join(expanded)

# ==================== K-Means Clustering ====================
N      = 10
true_k = 5

print("Grouping animals with K-Means Clustering...")
kmeans_model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=10, random_state=0)
kmeans_model.fit(tfidf_matrix)
df['Cluster_ID'] = kmeans_model.labels_

order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
terms           = vectorizer.get_feature_names_out()
print("Top terms per cluster:")
for i in range(true_k):
    top_words = [terms[ind] for ind in order_centroids[i, :7]]
    print(f"   Cluster {i+1}: {', '.join(top_words)}")
print("=" * 60)

# ==================== Search Function ====================
def search_animals(query, top_n=N):
    expanded_q = expand_query(query)
    if expanded_q != query.lower():
        print(f"\n   [Query Expansion] : '{expanded_q}'")

    query_vec    = vectorizer.transform([expanded_q])
    scores       = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices  = scores.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            'Index':   idx,
            'Name':    df.iloc[idx]['Animal Name'],
            'Score':   scores[idx],
            'URL':     df.iloc[idx]['Animal URL'],
            'Cluster': df.iloc[idx]['Cluster_ID'] + 1
        })
    return results

# ==================== Interactive Search Loop ====================
session_history = []

while True:
    print("\nWelcome to Animal Finder 🐾")
    print("(type 'q' or 'exit' to quit)")
    user_query = input("\n  Enter your search query: ")

    if user_query.lower() in ['q', 'exit', 'quit']:
        print("\nThank you for using Animal Finder 🐾")
        break

    if not user_query.strip():
        print("Please enter a valid search query.")
        continue

    clean_text  = re.sub(r'[^a-z\s]', ' ', user_query.lower())
    clean_words = [lemmatizer.lemmatize(w) for w in clean_text.split() if w not in stop_words]
    clean_query = " ".join(clean_words)

    if not clean_query.strip():
        print("   Please enter a valid animal trait or habitat.")
        continue

    if clean_query != user_query.lower():
        print(f"   Clean query : '{clean_query}'")

    # แสดง Inverted Index
    print("\n" + "-" * 60)
    print(f" Inverted Index : '{clean_query}'")
    print("-" * 60)
    print(f"{'Term':<15} | {'DF':<5} | {'Postings List'}")
    print("-" * 60)
    for term in clean_query.split():
        if term in inverted_index:
            df_count     = inverted_index[term]['DF']
            postings     = inverted_index[term]['Postings'][:10]
            postings_str = ", ".join(postings)
            if df_count > 10:
                postings_str += ", ..."
            print(f"{term:<15} | {df_count:<5} | [{postings_str}]")
        else:
            print(f"{term:<15} | 0     | [Not Found]")
    print("-" * 60)

    # ค้นหา
    results = search_animals(clean_query, top_n=N)

    if results:
        print(f"\n  Found {len(results)} animals matching '{user_query}'")
        for i, r in enumerate(results):
            print(f"[{i+1}] {r['Name']} ({r['URL']})")
            print(f"     Similarity: {r['Score']:.4f} | Cluster: {r['Cluster']}")

        # Evaluation
        expanded_eval = expand_query(clean_query)
        eval_pattern  = r'\b(?:' + '|'.join(re.escape(w) for w in expanded_eval.split()) + r')\b'

        relevant_docs = df[
            df['Cleaned Name'].str.contains(eval_pattern, case=False, na=False) |
            df['Cleaned Description'].str.contains(eval_pattern, case=False, na=False) |
            df['Cleaned Habitat'].str.contains(eval_pattern, case=False, na=False) |
            df['Cleaned Diet'].str.contains(eval_pattern, case=False, na=False) |
            df['Cleaned Behavior'].str.contains(eval_pattern, case=False, na=False) |
            df['Cleaned Physical'].str.contains(eval_pattern, case=False, na=False)
        ].index.tolist()

        retrieved_indices = [r['Index'] for r in results]
        tp_set = set(retrieved_indices[:N]).intersection(set(relevant_docs))
        TP = len(tp_set)
        FP = len(retrieved_indices[:N]) - TP
        FN = len(relevant_docs) - TP

        p_score  = precision_at_k(retrieved_indices, relevant_docs, N)
        r_score  = recall_at_k(retrieved_indices, relevant_docs, N)
        ap_score = average_precision(retrieved_indices, relevant_docs)

        session_history.append({'query': user_query, 'p': p_score, 'r': r_score, 'ap': ap_score})

        hits, sum_p = 0, 0.0
        for i, idx in enumerate(retrieved_indices):
            rank = i + 1
            if idx in relevant_docs:
                hits      += 1
                p_at_rank  = hits / rank
                sum_p     += p_at_rank
                print(f"   Rank {rank}: Match! -> P_{rank} = {hits}/{rank} = {p_at_rank:.2f}")
            else:
                print(f"   Rank {rank}: No match")

        total_relevant = len(relevant_docs)
        if total_relevant > 0:
            print(f"\n   Sum of Precisions = {sum_p:.2f}")
            print(f"   Total Relevant    = {total_relevant} items")
            print(f"   AP = {sum_p:.2f} / {total_relevant} = {ap_score:.2f}")
        else:
            print(f"\n   AP = 0.00")

        print(f"   [+] True Positive  (TP) : {TP}")
        print(f"   [-] False Positive (FP) : {FP}")
        print(f"   [-] False Negative (FN) : {FN}")
        print("\n" + "=" * 60)
        print(f"Precision_{N}: {p_score:.2f} | Recall_{N}: {r_score:.2f} | AP: {ap_score:.2f}")
        print("=" * 60)

        # Session history
        print(f"\n{'Query':<20} | {'Precision':<10} | {'Recall':<10} | {'Avg Precision'}")
        print("-" * 60)
        ap_list = []
        for h in session_history:
            print(f"{h['query']:<20} | {h['p']:<10.2f} | {h['r']:<10.2f} | {h['ap']:.2f}")
            ap_list.append(h['ap'])
        print("-" * 60)
        print(f"MAP (Mean Average Precision) : {np.mean(ap_list):.2f}")

    else:
        print(f"\n  No results found for '{user_query}'")

    print("-" * 60)