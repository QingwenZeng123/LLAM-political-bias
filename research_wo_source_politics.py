import guidance
import pandas as pd
from guidance import models, select
import csv


@guidance
def qa_bot(lm, query):
    lm += f'''\
    Q: {query}
    '''
    return lm


def get_llm_answers(data, model_name, context):
    """
    This is a method to send the prompt to llm and get answers

    :data: a dataframe used to store prompt and candidats information
    :model_name: a string to mention what llm used, need to initiate a new model every time ask questions 
    :context: is a string that provide llm some backgroud information, eg: the definition of bias
    :return: answers is an array store 'is-biased' or "is-not-biased", and array index is same as article id
    """
    answers = [''] * len(data)
    # Jump over n_ctx article id: [43]
    for index, row in data.iterrows():
        # while this article id haven't been evaluated
        if 
        
        
        
        
        row['id'] not in [43] and answers[row['id']] == '':
            print('id is: ', row['id'], 'title is:', row['article_title'])
            query = context + row['prompt'] # This prompt consist of article title and content
            if model_name == 'llama':
                # Load the llama
                llama = models.LlamaCpp('orca_mini_v3_7b.Q4_K_M.gguf', 
                                         n_ctx=4096,  # Context window size 
                                         n_gpu_layers=-1,  # -1 to use all GPU layers, 0 to use only CPU
                                         verbose=False  # Whether to print debug info
                                        )
                lm = llama + qa_bot(query)
                lm += select(['is-biased', "is-not-biased"], name='answer')
                print(lm["answer"])
                answers[row['id']] = str(lm['answer'])
    print("Predict from", model_name, "without any attributes is :")
    print(answers)
    return answers



# Loading data from excel file
data = pd.read_csv('data/data_prompt_w_articles_info_only.csv')

# Background context without any attributes
context_wo_info = "You are going to be the reader of a political article. Your job is to determine whether the article is biased. You answer either is-biased or is-not-biased, with no explanation. An article is biased in its presentation of the topic, meaning that it ever exaggerates, misrepresents, omits, or otherwise distorts facts (including by making subjective opinions look like facts) for the purpose of appealing to a certain political group."

# call the get_llm_answers to get predict answers
wo_info_answers = get_llm_answers(data, "llama", context_wo_info)

# Output the result to csv file: result_data/wo_attriute.csv
filename = "result_data/wo_attribute.cvs"
with open(filename, mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['id', 'wo_result'])

    # Write the answers to the CSV file
    for index in range(len(wo_info_answers)):
        if wo_info_answers[index] == '' and index != 43:
            break
        row = [index, wo_info_answers[index]]
        writer.writerow(row)

print(f'Data has been written to {filename}.')