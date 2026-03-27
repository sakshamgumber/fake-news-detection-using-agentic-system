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

    search_query = f"site:en.wikipedia.org {query}"
    
    payload = {
        "query": search_query,
        "num_results": num_results,
        "summary": True, # Get full summaries for richer reranking context

        # ✅ Hybrid search: combines semantic (vector) + keyword (BM25) search
        "search_type": "hybrid",
        "semantic_weight": 0.5,
        "keyword_weight": 0.5,

        # ✅ Restrict to English Wikipedia strictly
        "include_domains": ["en.wikipedia.org"],

        # ✅ English results only
        "language": "en",
        "country": "US"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Search API Error: {e}")
        return {}

def rerank_results(query: str, search_results: list):
    """
    Rerank search results using the LangSearch Reranker API.
    """
    
    url = "https://api.langsearch.com/v1/rerank"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Extract summaries to provide rich context for the reranker
    # Fallback to snippet if summary is missing (though summary=True is set)
    documents = [res.get("summary") or res.get("snippet", "") for res in search_results]
    documents = [doc for doc in documents if doc] # Filter out empty strings
    
    payload = {
        "model": "langsearch-reranker-v1",
        "query": query,
        "documents": documents,
        "top_n": 1,
        "return_documents": True,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        reranked_data = response.json()
        # Mapping reranked indices back to original results
        indices = [item["index"] for item in reranked_data.get("results", [])]
        return [search_results[i] for i in indices]
    except Exception as e:
        print(f"⚠️ Reranking failed: {e}. Falling back to original results.")
        return search_results

def search(query: str, num_results: int = 10):
    """
    Complete search pipeline: 
    1. Hybrid search on Wikipedia
    2. Filter to ensure Wikipedia domains
    3. Rerank for optimal relevance
    """
    # 1. Initial Search
    data = langsearch_wikipedia(query=query, num_results=num_results)
    
    results = data.get("data", {}).get("webPages", {}).get("value", [])
    
    # 1a. Post-filter to ensure results are strictly from Wikipedia
    results = [res for res in results if "wikipedia.org" in res.get("url", "")]
    
    if not results:
        return []

    # 2. Rerank for best relevance
    reranked_results = rerank_results(query, results)
    
    return reranked_results

def main():
    # Use command line argument if provided, otherwise use default query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "which is the highest peak of world"

    # Perform the full search & rerank pipeline
    reranked_results = search(query, num_results=10)
    
    if not reranked_results:
        print("⚠️ No reliable Wikipedia results found.")
        return

    # 3. Output only URLs
    print(f"\n✅ Top Reranked Wikipedia URLs for result:\n")
    for i, item in enumerate(reranked_results, start=1):
        url = item.get("url")
        if url:
            print(f"{i}. {url}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()


