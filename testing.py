# Debug: Check what endpoints are available
import requests
ngrok_url = "https://salvatore-selectable-overcleanly.ngrok-free.dev"

response = requests.get(f"{ngrok_url}/api/tags")
print("Available endpoints:", response.json())