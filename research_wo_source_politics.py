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

 # Load the llama
llama = models.LlamaCpp('orca_mini_v3_7b.Q4_K_M.gguf', 
                            n_ctx=4096,  # Context window size 
                            n_gpu_layers=-1,  # -1 to use all GPU layers, 0 to use only CPU
                            verbose=False  # Whether to print debug info
                        )


def get_llm_answers(data, model_name, context, prompt_column_name):
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
        if prompt_column_name == 'prompt_article_info' and row['article_id'] not in [43] and answers[row['id']] == '':
            print('id is: ', row['article_id'])
            query = context + row[prompt_column_name] # This prompt consist of article title and content
            if model_name == 'llama':
                lm = llama + qa_bot(query)
                lm += select(["is-not-biased", "is-biased"], name='answer')
                print(lm["answer"])
                answers[row['article_id']] = str(lm['answer'])
        elif prompt_column_name != 'prompt_article_info' and row['article_id'] != 43:
            query = context + row[prompt_column_name] # This prompt consist of article title and content
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
                answers[index] = str(lm['answer'])
    print("Predict from", model_name, "without any attributes is :")
    print(answers)
    return answers

def run_experiment_and_write_csv(data, model_name, context, output_filename, prompt_column_name):
    # Call the function to get llama answers
    answers = get_llm_answers(data, model_name, context, prompt_column_name)

    # Output the result to a CSV file
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'result'])

        # Write the answers to the CSV file
        for index, answer in enumerate(answers):
            if prompt_column_name == 'prompt_article_info' and answer == '' and index != 43:
                break
            row = [index, answer]
            writer.writerow(row)

data = pd.read_csv('data/prompts.csv')
# Background context without any attributes\t
context_wo_info = "You are going to be the reader of a political article. Your job is to determine whether the article is biased. You answer either is-biased or is-not-biased, with no explanation. An article is-biased in its presentation of the topic, meaning that it ever exaggerates, misrepresents, omits, or otherwise distorts facts (including by making subjective opinions look like facts) for the purpose of appealing to a certain political group.\n"

def run_experiments():
    #run_experiment_and_write_csv(data, 'llama', context_wo_info, 'result_data/testing.csv', 'nothing')
    #run_experiment_and_write_csv(data, "llama", context_wo_info, "result_data/original_prompt_result2.csv", 'prompt_article_info')
    run_experiment_and_write_csv(data, "llama", context_wo_info, "result_data/participant_politics_result.csv", "prompt_politics_info")
    run_experiment_and_write_csv(data, "llama", context_wo_info, "result_data/article_source_result.csv", "prompt_letter_source_info")
    run_experiment_and_write_csv(data, "llama", context_wo_info, "result_data/all_result.csv", "prompt_all_info")

run_experiments()