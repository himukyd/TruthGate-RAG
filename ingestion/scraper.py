import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urlparse
from tqdm import tqdm
import xml.etree.ElementTree as ET

BASE_URL = "https://fastapi.tiangolo.com"
SITEMAP_URL = "https://fastapi.tiangolo.com/sitemap.xml"
OUTPUT_DIR = "data/raw_docs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_urls_from_sitemap():
    print(f"Fetching sitemap from {SITEMAP_URL}...")
    try:
        response = requests.get(SITEMAP_URL, timeout=10)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        # Sitemaps use namespaces
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        urls = []
        for url_tag in root.findall('ns:url', namespace):
            loc_tag = url_tag.find('ns:loc', namespace)
            if loc_tag is not None:
                url = loc_tag.text
                # Only keep English documentation (skip /fr/, /es/, etc.)
                parsed = urlparse(url)
                path_parts = parsed.path.strip("/").split("/")
                
                # Check if it's a non-English language path (e.g., /fr/, /zh/)
                # Most language paths are 2 characters
                if len(path_parts[0]) == 2 and path_parts[0] != "en":
                    continue
                    
                urls.append(url)
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

def scrape_page(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Target the main content area
        content = soup.select_one("article") or soup.select_one(".md-content__inner")
        
        if not content:
            return None, None

        # Clean up
        for unwanted in content.select(".md-content__button, .headerlink, .md-clipboard"):
            unwanted.decompose()

        title = soup.title.string.split("-")[0].strip() if soup.title else "No Title"
        text = content.get_text(separator="\n", strip=True)
        
        return title, text
    except Exception as e:
        # Silently fail for specific pages to keep logs clean
        return None, None

def get_urls_from_page(url_to_crawl):
    print(f"Crawling {url_to_crawl} for additional links...")
    try:
        response = requests.get(url_to_crawl, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        urls = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/"):
                url = BASE_URL + href
            elif href.startswith(BASE_URL):
                url = href
            else:
                continue
                
            # Clean URL (remove anchors and queries)
            url = url.split("#")[0].split("?")[0].rstrip("/")
            
            # Filter non-English
            parsed = urlparse(url)
            path_parts = parsed.path.strip("/").split("/")
            if path_parts and len(path_parts[0]) == 2 and path_parts[0] != "en":
                continue
            
            if url.startswith(BASE_URL):
                urls.add(url)
        return urls
    except Exception as e:
        print(f"Error crawling {url_to_crawl}: {e}")
        return set()

def main():
    sitemap_urls = get_urls_from_sitemap()
    all_urls = set(sitemap_urls)
    
    seeds = [
        BASE_URL,
        f"{BASE_URL}/tutorial/",
        f"{BASE_URL}/advanced/",
        f"{BASE_URL}/how-to/",
        f"{BASE_URL}/reference/"
    ]
    
    for seed in seeds:
        all_urls.update(get_urls_from_page(seed))
    
    if not all_urls:
        print("Failed to retrieve any URLs.")
        return

    print(f"Found {len(all_urls)} potential English documentation pages.")
    
    # We want to reach at least 200
    target_urls = sorted(list(all_urls))

    for url in tqdm(target_urls, desc="Scraping Pages"):
        if len(os.listdir(OUTPUT_DIR)) >= 300:
            break
            
        title, text = scrape_page(url)
        if not text or len(text) < 200: # Lowered threshold
            continue

        # Create a safe filename
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            filename = "index.txt"
        else:
            filename = path.replace("/", "_") + ".txt"

        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            continue

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Source: {url}\n")
            f.write(f"Title: {title}\n\n")
            f.write(text)
        
        # Respectful delay
        time.sleep(0.05) # Slightly faster

    print(f"Successfully scraped {len(os.listdir(OUTPUT_DIR))} pages.")

if __name__ == "__main__":
    main()
