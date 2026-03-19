"""
LLM Interface - Unified interface for Ollama (free) and OpenAI (paid) LLMs
Provides automatic fallback and retry logic for robust operation
"""

import os
import time
from typing import Optional, Dict, Any, List
from enum import Enum
import yaml
from pathlib import Path
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import json
from dotenv import load_dotenv

# Load .env file so API keys are available via os.environ
load_dotenv()


class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"


class LLMInterface:
    """
    Unified interface for multiple LLM providers with fallback support.

    Usage:
        llm = LLMInterface()
        response = llm.generate("What is the capital of France?")
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize LLM interface with configuration.

        Args:
            config_path: Path to api_config.yaml file
        """
        self.config_path = config_path or self._find_config()
        self.config = self._load_config()

        # Initialize providers
        self.ollama_client = None
        self.openai_client = None
        self.gemini_client = None
        self.groq_client = None

        self._setup_providers()

    def _find_config(self) -> str:
        """Find configuration file in project structure"""
        possible_paths = [
            Path("config/api_config.yaml"),
            Path("../config/api_config.yaml"),
            Path("../../config/api_config.yaml"),
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        logger.warning("api_config.yaml not found, using defaults")
        return None

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path or not Path(self.config_path).exists():
            return self._default_config()

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Replace environment variables
        config = self._replace_env_vars(config)
        return config

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'llm': {
                'primary_provider': 'ollama',
                'fallback_provider': None,
                'ollama': {
                    'base_url': 'http://localhost:11434',
                    'model': 'llama3.2:3b',
                    'timeout': 60,
                    'temperature': 0.3,
                    'max_tokens': 2048
                },
                'groq': {
                    'model': 'openai/gpt-oss-120b',
                    'timeout': 60,
                    'temperature': 0.3,
                    'max_tokens': 2048
                },
                'gemini': {
                    'model': 'gemini-2.0-flash',
                    'timeout': 60,
                    'temperature': 0.3,
                    'max_tokens': 2048
                }
            }
        }

    def _replace_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Replace ${VAR} with environment variable values"""
        config_str = json.dumps(config)

        # Find all ${VAR} patterns
        import re
        pattern = r'\$\{([^}]+)\}'

        for match in re.finditer(pattern, config_str):
            var_name = match.group(1)
            var_value = os.getenv(var_name, '')
            config_str = config_str.replace(f'${{{var_name}}}', var_value)

        return json.loads(config_str)

    def _setup_providers(self):
        """Initialize LLM provider clients"""
        llm_config = self.config.get('llm', {})
        primary_provider = llm_config.get('primary_provider')
        fallback_provider = llm_config.get('fallback_provider')

        # Always setup Ollama, Gemini, and Groq since agents use them directly
        self._setup_ollama()
        self._setup_gemini()
        self._setup_groq()

    def _setup_ollama(self):
        """Initialize Ollama client using OpenAI API"""
        try:
            from openai import OpenAI
            
            base_url = self.config.get('llm', {}).get('ollama', {}).get('base_url', 'http://localhost:11434')
            if not base_url.endswith('/v1'):
                base_url = f"{base_url.rstrip('/')}/v1"
                
            self.ollama_client = OpenAI(
                base_url=base_url,
                api_key="ollama",  # API key is required by the client but unused by local Ollama
            )
            logger.info("Ollama client initialized successfully via OpenAI API")
        except ImportError:
            logger.warning("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")

    def _setup_openai(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI

            api_key = self.config['llm']['openai'].get('api_key')
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("OpenAI API key not provided")
        except ImportError:
            logger.warning("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")

    def _setup_groq(self):
        """Initialize Groq client"""
        try:
            from openai import OpenAI

            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                self.groq_client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
                logger.info("Groq client initialized successfully")
            else:
                logger.warning("GROQ_API_KEY not found in environment variables")
        except ImportError:
            logger.warning("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {e}")

    def _setup_gemini(self):
        """Initialize Google Gemini client via OpenAI-compatible API"""
        try:
            from openai import OpenAI

            api_key = os.environ.get("GEMINI_API_KEY")
            logger.info(f"Gemini API key: {api_key}")
            if api_key:
                self.gemini_client = OpenAI(
                    api_key=api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                )
                logger.info("Gemini client initialized successfully")
            else:
                logger.warning("GEMINI_API_KEY not found in environment variables")
        except ImportError:
            logger.warning("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        provider: Optional[LLMProvider] = None
    ) -> str:
        """
        Generate text using configured LLM with automatic fallback.

        Args:
            prompt: User prompt/query
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            provider: Force specific provider (None = auto)

        Returns:
            Generated text response

        Raises:
            Exception if all providers fail
        """
        llm_config = self.config.get('llm', {})

        # Determine provider order
        if provider:
            providers = [provider.value]
        else:
            providers = [llm_config.get('primary_provider', 'ollama')]
            fallback = llm_config.get('fallback_provider')
            if fallback:
                providers.append(fallback)

        # Try each provider in order
        last_error = None
        for prov_name in providers:
            try:
                if prov_name == 'ollama' and self.ollama_client:
                    return self._generate_ollama(prompt, max_tokens)
                elif prov_name == 'openai' and self.openai_client:
                    return self._generate_openai(prompt, max_tokens)
                elif prov_name == 'groq' and self.groq_client:
                    return self._generate_groq(prompt, max_tokens)
                elif prov_name == 'gemini' and self.gemini_client:
                    return self._generate_gemini(prompt, max_tokens)
            except Exception as e:
                logger.warning(f"{prov_name} generation failed: {e}")
                last_error = e
                continue

        # All providers failed
        raise Exception(f"All LLM providers failed. Last error: {last_error}")

    def _generate_ollama(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        role: str = 'user'
    ) -> str:
        """Generate text using Ollama via OpenAI API"""
        ollama_config = self.config['llm']['ollama']

        messages = [{'role': role, 'content': prompt}]

        response = self.ollama_client.chat.completions.create(
            model=model or ollama_config.get('model', 'llama3.2:3b'),
            messages=messages,
            temperature=temperature or ollama_config.get('temperature', 0.3),
            max_tokens=max_tokens or ollama_config.get('max_tokens', 2048)
        )
        return response.choices[0].message.content

    def _generate_openai(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        role: str = 'user'
    ) -> str:
        """Generate text using OpenAI"""
        openai_config = self.config['llm']['openai']

        messages = [{'role': role, 'content': prompt}]

        response = self.openai_client.chat.completions.create(
            model=model or openai_config.get('model', 'gpt-4o-mini'),
            messages=messages,
            temperature=temperature or openai_config.get('temperature', 0.3),
            max_tokens=max_tokens or openai_config.get('max_tokens', 2048)
        )

        return response.choices[0].message.content

    def _generate_groq(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        role: str = 'user'
    ) -> str:
        """Generate text using Groq"""
        groq_config = self.config['llm']['groq']

        messages = [{'role': role, 'content': prompt}]

        response = self.groq_client.chat.completions.create(
            model=model or groq_config.get('model', 'openai/gpt-oss-120b'),
            messages=messages,
            temperature=temperature or groq_config.get('temperature', 0.3),
            max_tokens=max_tokens or groq_config.get('max_tokens', 2048)
        )

        return response.choices[0].message.content

    def _generate_gemini(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        role: str = 'user'
    ) -> str:
        """Generate text using Google Gemini via OpenAI API"""
        gemini_config = self.config['llm'].get('gemini', {})

        messages = [{'role': role, 'content': prompt}]

        response = self.gemini_client.chat.completions.create(
            model=model or gemini_config.get('model', 'gemini-2.0-flash'),
            messages=messages,
            temperature=temperature or gemini_config.get('temperature', 0.3),
            max_tokens=max_tokens or gemini_config.get('max_tokens', 2048)
        )

        return response.choices[0].message.content

    
# Convenience function for quick usage
def get_llm() -> LLMInterface:
    """Get configured LLM interface singleton"""
    if not hasattr(get_llm, '_instance'):
        get_llm._instance = LLMInterface()
    return get_llm._instance
