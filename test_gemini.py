from google import genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Define the model and content
model = "gemini-embedding-001"
content = ["What is the capital of France?"]

# Request embeddings
response = client.models.embed_content(model=model, contents=content)

# Output the embeddings
print("Embedding:", response.embeddings)
