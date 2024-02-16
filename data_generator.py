import pandas as pd
import random
import requests
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm
import chardet

def exporting_data_to_csv(data, file_name):
    data.to_csv(file_name, index= False)

def preprocessing_original_data_file():
    data = pd.read_csv('news_bias_full_data.csv')


    clean_data = data[['Answer.age', 'Answer.articleNumber', 'Answer.batch', 'Answer.bias-question', 'Answer.country', 'Answer.gender', 'Answer.language1', 'Answer.newsOutlet', 'Answer.politics', 'Answer.url']]
    clean_data.rename(columns = {'Answer.age':'age', 'Answer.articleNumber':'articleNumber',
                                  'Answer.batch':'batch', 'Answer.bias-question': 'bias-question', 'Answer.country': 'country', 'Answer.gender': 'gender', 'Answer.language1': 'language', 'Answer.newsOutlet': 'source', 'Answer.politics': 'politics', 'Answer.url': 'url',}, inplace = True)
    exporting_data_to_csv(clean_data, "data/clean_original_data.csv")
    
def get_article_details(url):
  try:
    response = requests.get(url)

    # Detect the encoding of the webpage
    detected_encoding = chardet.detect(response.content)['encoding']

    # Check if the detected encoding matches with response's encoding
    if detected_encoding != response.encoding:
        print(f"Detected Encoding: {detected_encoding}, Response Encoding: {response.encoding}")

    # If the detected encoding exists, set it as the response's encoding
    if detected_encoding:
        response.encoding = detected_encoding
    else:
        # If no encoding was detected, then default to 'utf-8'
        response.encoding = 'utf-8'

    if response.status_code != 200:
        print(
            f"Failed to fetch the article URL ({url}). Status code: {response.status_code}")
        return None, None

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.find("h1").text.strip()
    content = []
    body = soup.find("article")
    if body:
        for paragraph in body.find_all("p"):
            content.append(paragraph.text.strip())

    return title, " ".join(content)
  except Exception as e:
      print(f"Error occurred: {e}")
      return None, None
  
def adding_articles_info_to_csv():
  
    data = pd.read_csv("data/clean_original_data.csv")

    # Create a new column 'id' that has unique IDs for each unique URL
    data['id'] = data['url'].factorize()[0]

    # Sort the dataframe by the 'id' column
    data = data.sort_values(by='id')

    # Reset the index to be in line with the new 'id' column
    data.reset_index(drop=True, inplace=True)
    data.drop(data[data['url'].isna()].index)

    count = 0

    id_mappings = {id_val: {'title': None, 'content': None} for id_val in data['id'].unique()}

    for index, row in data.iterrows():
      url = row['url']
      id = row['id']
      if id == count:
        article_details = get_article_details(url)
        id_mappings[id] = {'title': article_details[0], 'content': article_details[1]}
        count += 1
    
    # Create new columns based on the mappings
    data['article_title'] = data['id'].map(lambda x: id_mappings[x]['title'] if x in id_mappings else None)
    data['article_content'] = data['id'].map(lambda x: id_mappings[x]['content'] if x in id_mappings else None)
    
    exporting_data_to_csv(data, "data/data_articles_info.csv")
    

def creating_csv(csv_name):
    data = pd.read_csv('data/data_articles_info.csv')
    prompt_article_info = []
    prompt_reader_info = []
    prompt_politics_info = []
    prompt_letter_source_info = []
    prompt_all_info = []

    article_info = '''
    Article Title: {title}
    Article Content: {content}
    '''

    reader_info = "Reader Demographics: {age} years old {gender} {language} speaker from {country}\n"
    politics_info = "Reader identifies politically as: {politics}\n"
    letter_source_info = "Source of the letter: {letter_source}\n"

    for _, row in data.iterrows():
        article_info_question = article_info.format(
            title=row['article_title'],
            content=row['article_content']
        )
        
        reader_info_question = reader_info.format(
            age=row['age'],
            gender=row['gender'],
            language=row['language'],
            country=row['country']
        )
        
        letter_source_info_question = letter_source_info.format(
            letter_source=row['source']
        )
        
        politics_info_question = politics_info.format(
            politics=row['politics']
        )
        # Prompt with article_info only
        prompt_article_info.append(article_info_question)

        # Prompt with reader_info
        prompt_reader_info.append(reader_info_question + article_info_question)

        # Prompt with politics_info
        prompt_politics_info.append(politics_info_question + article_info_question)

        # Prompt with letter_source_info
        prompt_letter_source_info.append(letter_source_info_question + article_info_question)

        # Prompt with all_info
        prompt_all_info.append(reader_info_question + politics_info_question + letter_source_info_question + article_info_question)

    data['prompt_article_info'] = prompt_article_info
    data['prompt_reader_info'] = prompt_reader_info
    data['prompt_politics_info'] = prompt_politics_info
    data['prompt_letter_source_info'] = prompt_letter_source_info
    data['prompt_all_info'] = prompt_all_info

    exporting_data_to_csv(data, csv_name)

# Example usage:
creating_csv("data/prompts.csv")


# def creating_csv_prompt_w_article_details_only():
#     data = pd.read_csv("data/data_articles_info.csv")
#     # Prompt without reader's position
#     prompt_original = []

#     question = '''
#     Article Title: {title}
#     Article Content: {content}
#     '''

#     for index, row in data.iterrows():
#         info = question.format(
#             title=row['article_title'],  # Fixed typo in column name
#             content=row['article_content']  # Fixed typo in column name
#         )
#         prompt_original.append(info)

#     # Add 'prompt_without_sources_and_politics' column to the DataFrame
#     data['prompt'] = prompt_original
#     exporting_data_to_csv(data, "data/data_prompt_w_articles_info_only.csv")

# def creating_csv_prompt_wo_politics_source():
#     data = pd.read_csv("data/data_articles_info.csv")
#     # Prompt without reader's position
#     prompt_original = []

#     question = '''
#     Reader Demographics: {age} years old {gender} {language} speaker from {country}. Reader identifies politically as: {politics}
#     Article Title: {title}
#     Article Content: {content}
#     '''

#     for index, row in data.iterrows():
#         info = question.format(
#             age=row['age'],
#             gender=row['gender'],
#             language=row['language'],
#             country=row['country'],
#             politics=row['politics'],
#             title=row['article_title'],  # Fixed typo in column name
#             content=row['article_content']  # Fixed typo in column name
#         )
#         prompt_original.append(info)

#     # Add 'prompt_without_sources_and_politics' column to the DataFrame
#     data['prompt'] = prompt_original
#     exporting_data_to_csv(data, "data/data_prompt_wo_politics_sources.csv")


# adding_title_content_info_to_csv()
# creating_csv_prompt_wo_politics_source()
# creating_csv_prompt_w_article_details_only()