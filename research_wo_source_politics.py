import guidance
import pandas as pd
from guidance import models, select
import csv

llama = models.LlamaCpp('orca_mini_v3_7b.Q4_K_M.gguf', 
n_ctx=4096,  # Context window size
n_gpu_layers=-1,  # -1 to use all GPU layers, 0 to use only CPU
verbose=False  # Whether to print debug info
)


# Define the bias evaluation prompt
bias_evaluation_prompt = (
    "You are going to be the reader of a political article. Your job is to determine whether the article is biased based on the reader's perspective. You answer either biased or not biased, with no explanation. An article is biased in its presentation of the topic, meaning that it ever exaggerates, misrepresents, omits, or otherwise distorts facts (including by making subjective opinions look like facts) for the purpose of appealing to a certain political group."
)


@guidance
def qa_bot(lm, query):
    lm += f'''\
    Q: {query}
    '''
    return lm


# Loading data from excel file
data = pd.read_csv('data/data_prompt_w_articles_info_only.csv')
# This is an array to store the nobias/bias of article id = index
answers = [''] * len(data)

# Jump over n_ctx article id: [43]
for index, row in data.iterrows():
    # while this article id haven't been evaluated
    if row['id'] not in [43] and answers[row['id']] == '':
        print('index is: ', index, 'id is: ', row['id'])
        print(row['article_title'])
        query = row['prompt']
        lm = llama + qa_bot(query)
        lm += select(['bias', 'no bias'], name='answer')
        print(lm["answer"])
        answers[row['id']] = str(lm['answer'])
print("Final result is: ")
print(answers)

# Output the result to csv file: result_data/wo_attriute.csv
filename = "result_data/wo_attribute.cvs"
with open(filename, mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['id', 'wo_result'])

    # Write the answers to the CSV file
    for index in range(len(answers)):
        if answers[index] == '' and index != 43:
            break
        row = [index, answers[index]]
        writer.writerow(row)

print(f'Data has been written to {filename}.')