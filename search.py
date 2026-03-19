import requests
import sys


API_KEY = "sk-8ef39cbed62941f7a0af9dcb67c09752"

def langsearch_wikipedia(query: str, num_results: int = 10):
    """
    Perform a hybrid search restricted to Wikipedia in English.
    """
    url = "https://api.langsearch.com/v1/web-search"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "query": query,
        "num_results": num_results,
        "summary": False, # Only need results for reranking and final URL output

        # ✅ Hybrid search: combines semantic (vector) + keyword (BM25) search
        "search_type": "hybrid",
        "semantic_weight": 0.5,
        "keyword_weight": 0.5,

        # ✅ Restrict to Wikipedia domain only
        "include_domains": ["wikipedia.org"],

        # ✅ English results only
        "language": "en",
        "country": "US"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(response.json())
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Search API Error: {e}")
        return {}

def rerank_results(query: str, search_results: list):
    """
    Rerank search results using the LangSearch Reranker API.
    """
    if not search_results:
        return []

    url = "https://api.langsearch.com/v1/rerank"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Extract snippets to provide context for the reranker
    documents = [res.get("snippet", "") for res in search_results if res.get("snippet")]
    
    if not documents:
        return search_results

    payload = {
        "model": "langsearch-reranker-v1",
        "query": query,
        "documents": documents,
        "top_n": len(documents)
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        print(response.json())
        response.raise_for_status()
        reranked_data = response.json()
        # Mapping reranked indices back to original results
        indices = [item["index"] for item in reranked_data.get("results", [])]
        return [search_results[i] for i in indices]
    except Exception as e:
        print(f"⚠️ Reranking failed: {e}. Falling back to original results.")
        return search_results

def main():
    # Use command line argument if provided, otherwise use default query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "which is the highest peak of world"

    print(f"\n🔎 Searching Wikipedia (Hybrid) for: '{query}'...")
    
    # 1. Initial Search
    data = langsearch_wikipedia(query=query, num_results=10)
    results = data.get("results", [])

    if not results:
        print("⚠️ No results found.")
        return

    # 2. Rerank for best relevance
    print("🔄 Reranking results for optimal accuracy...")
    reranked_results = rerank_results(query, results)

    # 3. Output only URLs
    print(f"\n✅ Top Reranked Wikipedia URLs for result:\n")
    for i, item in enumerate(reranked_results, start=1):
        url = item.get("url")
        if url:
            print(f"{i}. {url}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()


