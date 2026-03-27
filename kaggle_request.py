import requests
import os
from dotenv import load_dotenv

load_dotenv()

NGROK_URL = os.getenv("NG_ROCK_URL", "")


import json
import requests


import time

def query_model_with_requests(messages: list[dict], NGROK_PORT: str = NGROK_URL) -> str:
    # Convert messages list to a single prompt string
    prompt = "\n".join([f"{m.get('role', '')}: {m.get('content', '')}" for m in messages])

    data = {
        "model": "qwen3.5:4b",
        "prompt": prompt,
        "stream": False,
    }

    start_time = time.time()
    try:
        response = requests.post(f"{NGROK_PORT}/api/generate", json=data, stream=True)
        
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    full_response += chunk.get('response', '')
                    if chunk.get('done'):
                        break
            
            elapsed_time = time.time() - start_time
            print(f"Query completed in {elapsed_time:.2f} seconds")
            return full_response
        else:
            print(f"Query failed: {response.status_code}")
            return ""
    except requests.exceptions.RequestException as e:
        print(f"Error querying model with requests: {e}")
        return ""

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum entanglement simply."}
    ]
    print(query_model_with_requests(messages))