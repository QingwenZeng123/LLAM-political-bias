import guidance
import pandas as pd
from guidance import models, select

llama = models.LlamaCpp('orca_mini_v3_7b.Q4_K_M.gguf', 
n_ctx=4096,  # Context window size
n_gpu_layers=-1,  # -1 to use all GPU layers, 0 to use only CPU
verbose=True  # Whether to print debug info
)


# Define the bias evaluation prompt
bias_evaluation_prompt = (
    "You are going to be the reader of a political article. Your job is to determine whether the article is biased based on the reader's perspective. You answer either biased or not biased, with no explanation. "
)


@guidance
def qa_bot(lm, query):
    lm += f'''\
    Q: {query}
    '''
    return lm


answers = []

# Loading data from excel file
data = pd.read_csv('data/data_prompt_w_articles_info_only.csv')
for index, row in data.iterrows():
    query = row['prompt']
    lm = llama + qa_bot(query)
    lm += select(['bias', 'no bias'], name='answer')
    print(lm["answer"])
    answers.append(str(lm['answer']))

# Issue need to fix: The prompt is too long, can't run 