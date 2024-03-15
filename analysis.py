import pandas as pd

def collect_analysis(result_file_path, no_other_source = False):
    result_data = pd.read_csv(result_file_path)
    original_data = pd.read_csv('data/data_articles_info.csv', usecols=['id', 'bias-question', 'politics', 'article_id'])
    result_data = result_data[(result_data['id'] != 43) & (result_data['id'] != -1)]
    if no_other_source:
        merged_data = pd.merge(result_data, original_data, left_on= 'id', right_on='article_id', how='left', validate='one_to_many')
    else:
        merged_data = pd.merge(result_data, original_data, on='id', how='left', validate='one_to_one')
        print(merged_data)
        merged_data = merged_data[merged_data['result'].notna()]

        
    
    # Collect overall data
    correct_predictions = merged_data[merged_data['result'] == merged_data['bias-question']]
    overall = len(correct_predictions) / len(merged_data)
    
    # Perform accuracy calculation for each politics group
    politics_accuracy = {}
    
    for politics_group in merged_data['politics'].unique():
        if not pd.isna(politics_group):
            # Filter data for the current politics group
            group_data = merged_data[merged_data['politics'] == politics_group]

            # Compare 'result' and 'bias-question' to calculate accuracy within the politics group
            correct_predictions = group_data[group_data['result'] == group_data['bias-question']]
            accuracy = len(correct_predictions) / len(group_data)

            # Store accuracy in the dictionary with the politics group as the key
            politics_accuracy[politics_group] = accuracy
    return (overall, politics_accuracy)


result_file_paths = ['result_data/v2/original_prompt_result.csv', 'result_data/v2/participant_politics_result.csv']
output_csv_path = 'result_data/v2/analysis_results.csv'
first_output = []
conservative = []
liberal = []
independent = []
other = []
default = []
for file in result_file_paths:
    if file == 'result_data/v2/original_prompt_result.csv':
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
    'File Name': result_file_paths,
    'Overall': first_output,
    'Conservative': conservative,
    'Liberal': liberal,
    'Independent': independent,
    'Other': other,
    'default': default }
 )
df.to_csv('result_data/v2/analysis_result.csv', index= False) 