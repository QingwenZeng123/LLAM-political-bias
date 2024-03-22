import openai
from dotenv import load_dotenv
import os

# Load environment variable from .env
load_dotenv()
# create a client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Are you a human"}],
    stream=True,
)

for chunk in stream:
    # print(chunk.choices[0])
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
