import requests

# Base URL of your running FastAPI app
BASE_URL = "http://127.0.0.1:8000"

def test_chat_api():

    payload = {
        "message": "Hello, I need infomation about my business policies.",
        "session_id": None 
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)

    # Print results
    if response.status_code == 200:
        data = response.json()
        print("AI Response:", data.get("response"))
        print("Session ID:", data.get("session_id"))
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    test_chat_api()
