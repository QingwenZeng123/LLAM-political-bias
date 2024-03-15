from textwrap import indent

from regex import F
import guidance
import pandas as pd
from guidance import models, select, gen
import csv


@guidance
def qa_bot(lm, query):
    # choose the answer between is-biased and is-not biased based on guidance
    response = select(["is-not-biased", "is-biased"], name='answer')
    
    # Append the selected response to the language model
    lm += f'''\
    Q: {query}
    A: {response}'''
    return lm

def get_llm_answers(data, model_name, context, prompt_column_name):
    """
    This is a method to send the prompt to llm and get answers

    :data: a dataframe used to store prompt and candidats information
    :model_name: a string to mention what llm used, need to initiate a new model every time ask questions 
    :context: is a string that provide llm some backgroud information, eg: the definition of bias
    :return: answers is an array store 'is-biased' or "is-not-biased", and array index is same as article id
    """
    answers = [''] * len(data)
    current_article_id = 0
    # Jump over n_ctx article id: [43]
    for index, row in data.iterrows():
       
        # while this article id haven't been evaluated
        if prompt_column_name == 'prompt_article_info' and row['article_id'] == current_article_id:
            current_article_id += 1
            if row['article_id'] == 43:
                continue
            print('id is: ', row['article_id'])
            query = context + row[prompt_column_name] # This prompt consist of article title and content
            if model_name == 'llama':
                # Load the llama
                llama = models.LlamaCpp('orca_mini_v3_7b.Q4_K_M.gguf', 
                            n_ctx=4096,  # Context window size 
                            n_gpu_layers=-1,  # -1 to use all GPU layers, 0 to use only CPU
                            verbose=False  # Whether to print debug info
                        )
                lm = llama + qa_bot(query)
                print(lm["answer"])
                answers[row['article_id']] = str(lm['answer'])
        elif prompt_column_name != 'prompt_article_info' and row['article_id'] != 43:
            print("index is", index)
            query = context + row[prompt_column_name] # This prompt consist of article title and content
            if model_name == 'llama':
                # Load the llama
                llama = models.LlamaCpp('orca_mini_v3_7b.Q4_K_M.gguf', 
                                            n_ctx=4096,  # Context window size 
                                            n_gpu_layers=-1,  # -1 to use all GPU layers, 0 to use only CPU
                                            verbose=False  # Whether to print debug info
                                        )
                lm = llama + qa_bot(query)
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
        if prompt_column_name == 'prompt_article_info':  
            writer.writerow(['article_id', 'result'])
        else:
            writer.writerow(['id', 'result'])

        # Write the answers to the CSV file
        for index, answer in enumerate(answers):
            if prompt_column_name == 'prompt_article_info' and answer == '' and index != 43:
                break
            row = [index, answer]
            writer.writerow(row)

data = pd.read_csv('data/prompts.csv')


def run_experiments():
    # Contexts for 4 experiments
    article_only_context = "You are going to be the reader of a political article. Your job is to determine whether the article is biased based on reader point of view. \
                              You answer either is-biased or is-not-biased, with no explanation. An article is-biased in its presentation of the topic, \
                              meaning that it ever exaggerates, misrepresents, omits, or otherwise distorts facts (including by making subjective opinions look like facts)\
                              for the purpose of appealing to a certain political group.\n"

    political_side_context = "You are going to be the reader of a political article. Your job is to determine whether the article is biased based on the Reader identifies politics group.\
                              You answer either is-biased or is-not-biased, with no explanation.\
                              An article is-biased in its presentation of the topic, meaning that it ever exaggerates, misrepresents, omits,\
                              or otherwise distorts facts (including by making subjective opinions look like facts) for the purpose of appealing to a certain political group.\n"


    article_source_context = "You are going to be the reader of a political article. Your job is to determine whether the article is biased. And take source of the article into consideration to provide answer.\
                              You answer either is-biased or is-not-biased, with no explanation.\
                              An article is-biased in its presentation of the topic, meaning that it ever exaggerates, misrepresents, omits,\
                              or otherwise distorts facts (including by making subjective opinions look like facts) for the purpose of appealing to a certain political group.\n"


    all_information_context = "You are going to be the reader of a political article. Your job is to determine whether the article is biased.\
                              And take all the anticipants information into consideration to provide answer,\
                              which include Reader Demographics information, Readers' politics group and article source\
                              You answer either is-biased or is-not-biased, with no explanation.\
                              An article is-biased in its presentation of the topic, meaning that it ever exaggerates, misrepresents, omits,\
                              or otherwise distorts facts (including by making subjective opinions look like facts) for the purpose of appealing to a certain political group.\n"
    
    conservative_pov_context = "You are going to be the reader of a political article who is identical as Conservative. Your job is to determine whether the article is biased based on Conservative reader point of view. \
                              You answer either is-biased or is-not-biased, with no explanation. An article is-biased in its presentation of the topic, \
                              meaning that it ever exaggerates, misrepresents, omits, or otherwise distorts facts (including by making subjective opinions look like facts)\
                              for the purpose of appealing to a certain political group.\n"
                              
    liberal_pov_context = "You are going to be the reader of a political article who is identical as Liberal. Your job is to determine whether the article is biased based on :iberal reader point of view. \
                              You answer either is-biased or is-not-biased, with no explanation. An article is-biased in its presentation of the topic, \
                              meaning that it ever exaggerates, misrepresents, omits, or otherwise distorts facts (including by making subjective opinions look like facts)\
                              for the purpose of appealing to a certain political group.\n"
                              
    Independence_pov_context = "You are going to be the reader of a political article who is identical as Independence for politics. Your job is to determine whether the article is biased based on Independence reader point of view. \
                              You answer either is-biased or is-not-biased, with no explanation. An article is-biased in its presentation of the topic, \
                              meaning that it ever exaggerates, misrepresents, omits, or otherwise distorts facts (including by making subjective opinions look like facts)\
                              for the purpose of appealing to a certain political group.\n"

    run_experiment_and_write_csv(data, "llama", article_only_context, "result_data/v2/original_prompt_result.csv", 'prompt_article_info')
    run_experiment_and_write_csv(data, "llama", political_side_context, "result_data/v2/participant_politics_result.csv", "prompt_politics_info")
    run_experiment_and_write_csv(data, "llama", article_source_context, "result_data/v2/article_source_result.csv", "prompt_letter_source_info")
    run_experiment_and_write_csv(data, "llama", all_information_context, "result_data/v2/all_result.csv", "prompt_all_info")
    
    politics_prompts = [conservative_pov_context, liberal_pov_context, Independence_pov_context]
    folder_version = 3
    for politic_prompt in politics_prompts:
        run_experiment_and_write_csv(data, "llama", politic_prompt, "result_data/v" + folder_version + "/original_prompt_result.csv", 'prompt_article_info')
        run_experiment_and_write_csv(data, "llama", politic_prompt, "result_data/v" + folder_version + "/participant_politics_result.csv", "prompt_politics_info")
        run_experiment_and_write_csv(data, "llama", politic_prompt, "result_data/v" + folder_version + "/article_source_result.csv", "prompt_letter_source_info")
        run_experiment_and_write_csv(data, "llama", politic_prompt, "result_data/v" + folder_version + "/all_result.csv", "prompt_all_info")
        folder_version += 1

run_experiments()