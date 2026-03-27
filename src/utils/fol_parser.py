from typing import Dict, Any, List
from loguru import logger
import yaml
import json
from pathlib import Path
from kaggle_request import query_model_with_requests

class FOLParser:
    """
    First-Order Logic parser for claim decomposition.

    Decomposes complex claims into atomic predicates:
    - P(entity, property, value)
    - Relation(entity1, entity2)
    """

    def __init__(self, llm_interface=None):
        """
        Initialize FOL parser.

        Args:
            llm_interface: Optional LLM interface for intelligent decomposition
        """
        self.llm = llm_interface
        self.prompts = self._load_prompts()

    def _generate_ollama(self, messages: list) -> str:
        """Delegate to LLMInterface._generate_ollama, extracting prompt from messages."""
        prompt = messages[0]["content"] if messages else ""
        return self.llm._generate_ollama(prompt=prompt)

    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML"""
        possible_paths = [
            Path("config/agent_prompt.yaml"),
            Path("../config/agent_prompt.yaml"),
            Path("../../config/agent_prompt.yaml"),
        ]

        for path in possible_paths:
            if path.exists():
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
        return {}


    def _decompose_with_llm(self, claim: str):
        """Use LLM for intelligent claim decomposition"""
        prompt = self.prompts.get('claim_decomposition', '').format(claim=claim)

        try:
            messages = [{"role": "user", "content": prompt}]
            response_text = query_model_with_requests(messages)
            logger.debug(f'resposne text is{response_text}')
            # Parse JSON safely (handles empty, markdown-fenced, or malformed responses)
            logger.info(type(response_text))
            json_response=json.loads(response_text)
            if isinstance(json_response, dict):
                subclaims = json_response.get('subclaims', [])
            else:
                subclaims = json_response
            return subclaims

        except Exception as e:
            logger.error(f"LLM decomposition failed: {e}")
            return [{
                'subclaim': claim,
                'predicate': f'Claim({claim})',
                'verifiability': 'VERIFIABLE'
            }]

    def facts(self, subclaim: str) -> Dict[str, Any]:
        """Classify a subclaim as VERIFIABLE or NON-VERIFIABLE"""
        prompt = self.prompts.get('subclaim_classification', '').format(claim=subclaim)
        try:
            messages = [{"role": "user", "content": prompt}]
            response_text = query_model_with_requests(messages)
            json_response = json.loads(response_text)

            return json_response


        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return {
                'classification': f'Error: {e}',
                'verifiability': 'VERIFIABLE'
            }