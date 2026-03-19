from langsearch import Client

# You need to install the langsearch client first:
# pip install langsearch

# It's recommended to set the API key as an environment variable
# LANGSEARCH_API_KEY="..."
# client = Client()
# For this example, we will hardcode it.
# Please replace "YOUR_LANGSEARCH_API_KEY" with your actual key.
client = Client(api_key="sk-8ef39cbed62941f7a0af9dcb67c09752")

def search_and_rerank_wikipedia_with_langsearch(query: str, limit: int = 10):
    """
    Searches Wikipedia for a given query using LangSearch, retrieves the top results,
    and then uses the rerank API to identify the best result.

    Args:
        query (str): The search query.
        limit (int): The number of results to retrieve.

    Returns:
        dict: The best reranked result, or an error message.
    """
    try:
        # Step 1: Search Wikipedia using LangSearch
        # Add "site:en.wikipedia.org" to focus the search on Wikipedia
        search_query = f"{query} site:en.wikipedia.org"
        print(f"Performing search with LangSearch query: '{search_query}'")
        
        search_results = client.search(
            query=search_query,
            limit=limit
        )

        if not search_results or not search_results.get('results'):
            return {"error": "No search results found."}

        print(f"Found {len(search_results['results'])} results.")
        
        # Extract documents for reranking
        documents_to_rerank = search_results['results']

        if not documents_to_rerank:
            return {"error": "No documents found in search results to rerank."}

        print("Reranking documents...")
        rerank_results = client.rerank(
            query=query,
            documents=documents_to_rerank,
        )

        if not rerank_results or not rerank_results.get('results'):
            return {"error": "Reranking did not return any results."}
            
        # The best result is the first one in the reranked list
        best_result = rerank_results['results'][0]
        
        return best_result

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    search_query = "AI agent frameworks"
    
    # Perform the search and rerank
    best_url_info = search_and_rerank_wikipedia_with_langsearch(search_query)

    if "error" in best_url_info:
        print(f"An error occurred: {best_url_info['error']}")
    else:
        print("\n--- Best Reranked Result (LangSearch) ---")
        print(f"URL: {best_url_info.get('url')}")
        print(f"Score: {best_url_info.get('score')}")
        print(f"Title: {best_url_info.get('title')}")
        print(f"Content Snippet: {best_url_info.get('snippet', 'N/A')}")
