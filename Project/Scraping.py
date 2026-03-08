from bs4 import BeautifulSoup
import requests
import csv
import random
import time
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
    'animals-that-start', 'scientific', 'class', 'location',
    'endangered', 'amphibians', 'birds', 'fish', 'mammals',
    'reptiles', 'invertebrates', 'insects', 'page', 'quiz', 'blog'
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