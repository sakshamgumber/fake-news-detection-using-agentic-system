"""
Evidence Seeking Agent - 3-stage evidence retrieval pipeline
Stage 1: Internet Search
Stage 2: Credibility Check
Stage 3: Content Extraction
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger
from datetime import datetime
import yaml
import os
import sys
import requests
sys.path.append('..')
from kaggle_request import query_model_with_requests
from src.utils.search import langsearch_wikipedia, rerank_results
from src.utils.web_scraper import WebScraper
from src.utils.credibility_checker import CredibilityChecker

@dataclass
class Evidence:
    """Single piece of evidence"""
    source_url: str
    source_name: str
    passage: str
    retrieved_at: str


@dataclass
class EvidenceResult:
    """Result from Evidence Seeking Agent"""
    subclaim_text: str
    queries: List[str]
    evidence: List[Evidence]
    total_sources: int

class EvidenceSeekingAgent:
    """
    Agent 3: Evidence Seeking

    Three-stage pipeline:
    1. Internet Search (DuckDuckGo/SerperAPI)
    2. Credibility Assessment (Heuristic/MBFC)
    3. Content Extraction (BeautifulSoup)
    """

    def __init__(self, config: Dict[str, Any] = None,llm_interface=None):
        """
        Initialize Evidence Seeking Agent.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize components
        self.scraper = WebScraper(self.config.get('scraping', {}))
        self.credibility = CredibilityChecker(self.config.get('credibility', {}))

        # Configuration
        self.max_results = self.config.get('max_search_results', 10)
        self.credibility_threshold = self.config.get('credibility_threshold', 'medium')
        self.max_passages = self.config.get('max_passages_per_source', 3)
        self.llm = llm_interface
        logger.info("Evidence Seeking Agent initialized")
        
        # Load prompts
        self.prompt_template = None
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), '../../config/agent_prompt.yaml')
            with open(prompt_path, 'r') as f:
                self.prompt_template = yaml.safe_load(f).get('evidence_seeking', '')
        except Exception as e:
            logger.warning(f"Could not load agent_prompt.yaml: {e}")

    def _generate_ollama(self, messages: list) -> str:
        """Delegate to LLMInterface._generate_ollama, extracting prompt from messages."""
        prompt = messages[0]["content"] if messages else ""
        return self.llm._generate_ollama(prompt=prompt)

    def process(self, query_results: List[Any]) -> List[EvidenceResult]:
        """
        Retrieve evidence for all queries.

        Args:
            query_results: List of QueryResult objects

        Returns:
            List of EvidenceResult objects
        """
        logger.info(f"Seeking evidence for {len(query_results)} subclaims")

        results = []
        evidence_id_counter = 0
        for qr in query_results:
            evidence_list = []
            for query in qr.queries:
                # Stage 1: Search
                search_results = self._search_web(query)

                # Stage 2 & 3: Check credibility and extract content
                for url in search_results:
                    try:

                        # Content extraction
                        content = self.scraper.scrape(url)
                        passages = []
                        # Use LLM for extraction if available
                        content_text = getattr(content, 'text', str(content))[:5000]

                        prompt = self.prompt_template.format(
                            query=query,
                            content=content_text
                        )
                        messages = [{"role": "user", "content": prompt}]
                        extracted = query_model_with_requests(messages)
                        if extracted and "None" not in extracted:
                            passages = [extracted]
                        for passage in passages:
                            evidence = Evidence(
                                source_url=url,
                                source_name=content.title or url,
                                passage=passage,
                                retrieved_at=datetime.utcnow().isoformat(),
                            )
                            evidence_list.append(evidence)
                            evidence_id_counter += 1

                    except Exception as e:
                        logger.error(f"Error processing {url}: {e}")
                        continue

            results.append(EvidenceResult(
                subclaim_text=qr.subclaim_text,
                queries=qr.queries,
                evidence=evidence_list,
                total_sources=len(evidence_list),
            ))


        return results

    def _search_web(self, query: str) -> List[str]:
        """
        Search the web using LangSearch Wikipedia + reranking.

        Args:
            query: Search query
            dataset: Dataset identifier (unused for now, kept for future routing)

        Returns:
            List of URLs
        """
        try:
            clean_query = query.replace(" site:wikipedia.org", "").strip()

            # Step 1: Hybrid search on Wikipedia via LangSearch
            data = langsearch_wikipedia(clean_query, num_results=5)
            results = data.get("data", {}).get("webPages", {}).get("value", [])
            results = [res for res in results if "wikipedia.org" in res.get("url", "")]

            if not results:
                return []

            # Step 2: Rerank for best relevance
            reranked = rerank_results(clean_query, results)

            urls = [res.get("url") for res in reranked if res.get("url")]
            logger.debug(f"Search results for '{query}': {urls}")
            return urls

        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []

    
