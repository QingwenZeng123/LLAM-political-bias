import openai
from dotenv import load_dotenv
import os

# Load environment variable from .env
load_dotenv()
# create a client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_answer(model_version, content):
    """
    This is a method to send one request to chatgpt model 
    
    :model_version: a string to show what version of chatgpt model will be used
    :content: string from prompts.csv
    :return: string answer
    """
    request = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Are you a human"}],
        stream=False,
    )
    answer = request.choices[0].message.content
    print(answer)
    return answer
    # for chunk in stream:
#     if chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0].delta.content, end="")


def get_answers_list(data, model_version, context, prompt_column_name):
    """
    This is a method to get answer list of each experiment

    :data: a dataframe used to store prompt and candidats information
    :model_version: chatgpt model version
    :context: is a string that provide llm some backgroud information, eg: the definition of bias
    :prompt_column_name: a string to express which experiment is running
    :return: answer_list is an array store 'is-biased' or "is-not-biased", and array index is same as article id
    """
    



