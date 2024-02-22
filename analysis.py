import pandas as pd

def collect_analysis(result_file_path, noOtherSource = False):
    result_data = pd.read_csv(result_file_path)
    result_data['result'] = result_data['result'].replace({'bias': 'is-biased', 'no bias': 'is-not-biased'})
    original_data = pd.read_csv('data\data_articles_info.csv', usecols=['id', 'bias-question', 'politics'])
    result_data = result_data[(result_data['id'] != 43) & (result_data['id'] != -1)]
    if noOtherSource:
        merged_data = pd.merge(result_data, original_data, on='id', how='left')
    else:
        merged_data = pd.concat([result_data, original_data], ignore_index=True)
    #return merged_data
    correct_predictions = merged_data[merged_data['result'] == merged_data['bias-question']]
    
    politics_accuracy = {}
    
    for politics_group in merged_data['politics'].unique():
        # Filter data for the current politics group
        group_data = merged_data[merged_data['politics'] == politics_group]

        # Compare 'result' and 'bias-question' to calculate accuracy within the politics group
        correct_predictions = group_data[group_data['result'] == group_data['bias-question']]
        accuracy = len(correct_predictions) / len(group_data)

        # Store accuracy in the dictionary with the politics group as the key
        politics_accuracy[politics_group] = accuracy
    return (len(correct_predictions) / len(merged_data), politics_accuracy)


# result_file_paths = ['result_data/original_prompt_result.csv', 'result_data/participant_politics_result.csv', 'result_data/article_source_result.csv', 'result_data/all_result.csv']
result_file_paths = ['result_data/original_prompt_result.csv']
output_csv_path = 'result_data/analysis_results.csv'
first_output = []
conservative = []
liberal = []
independent = []
other = []
default = []
for file in result_file_paths:
    if file == 'result_data/original_prompt_result.csv':
        result = collect_analysis(file, True)
    else:
        result = collect_analysis(file)
    
    first_output.append(result[0])
    conservative.append(result[1]['Conservative'])
    liberal.append(result[1]['Liberal'])
    independent.append(result[1]['Independent'])
    other.append(result[1]['Other'])
    default.append(result[1]['default'])

df = pd.DataFrame({
    'Overall': first_output,
    'Conservative': conservative,
    'Liberal': liberal,
    'Independent': independent,
    'Other': other,
    'default': default }
 )
df.to_csv('result_data/analysis_result.csv', index= False) 