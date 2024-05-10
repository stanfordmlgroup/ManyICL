import traceback
import os
from tqdm import tqdm 
import random
import pickle
import numpy as np
from LMM import GPT4VAPI, GeminiAPI

def work(model, num_shot_per_class, location, num_qns_per_round, test_df, demo_df, classes, class_desp, SAVE_FOLDER, dataset_name, detail='auto', file_suffix=''):
    """
    Run queries for each test case in the test_df dataframe using demonstrating examples sampled from demo_df dataframe.
    
    model[str]: the specific model checkpoint to use e.g. "Gemini1.5", "gpt-4-turbo-2024-04-09"
    num_shot_per_class[int]: number of demonstrating examples to include for each class, so the total number of demo examples equals num_shot_per_class*len(classes)
    location[str]: Vertex AI location e.g. "us-central1","us-west1", not used for GPT-series models
    num_qns_per_round[int]: number of queries to be batched in one API call
    test_df, demo_df [pandas dataframe]: dataframe for test cases and demo cases, see dataset/UCMerced/demo.csv as an example
    classes[list of str]: names of categories for classification, and this should match tbe columns of test_df and demo_df. 
    class_desp[list of str]: category descriptions for classification, and these are the actual options sent to the model
    SAVE_FOLDER[str]: path for the images
    dataset_name[str]: name of the dataset used
    detail[str]: resolution level for GPT4(V)-series models, not used for Gemini models
    file_suffix[str]: suffix for image filenames if not included in indexes of test_df and demo_df. e.g. ".png"
    """
    
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    EXP_NAME = f'{dataset_name}_{num_shot_per_class*len(classes)}shot_{model}_{num_qns_per_round}'
    
    if model.startswith('gpt'):
        api = GPT4VAPI(model=model, detail=detail)
    else:
        assert model=='Gemini1.5'
        api = GeminiAPI(location=location)
    print(EXP_NAME, f'test size = {len(test_df)}')
    
    # Prepare the demonstrating examples
    demo_examples = []
    for class_name in classes:
        num_cases_class = 0
        for j in demo_df[demo_df[class_name] == 1].itertuples():
            if num_cases_class == num_shot_per_class:
                break
            demo_examples.append((j.Index, class_desp[class_to_idx[class_name]]))
            num_cases_class += 1
    assert len(demo_examples) == num_shot_per_class*len(classes)

    #Load existing results
    if os.path.isfile(f'{EXP_NAME}.pkl'):
        with open(f'{EXP_NAME}.pkl', 'rb') as f:
            results = pickle.load(f)
    else:
        results = {}

    test_df = test_df.sample(frac=1, random_state=66) #Shuffle the test set
    for start_idx in tqdm(range(0, len(test_df), num_qns_per_round), desc=EXP_NAME):
        end_idx = min(len(test_df), start_idx+num_qns_per_round)
        
        random.shuffle(demo_examples)
        prompt = ""
        image_paths = [os.path.join(SAVE_FOLDER, i[0]+file_suffix) for i in demo_examples]
        for demo in demo_examples:
            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
Question: What is in the image above?
Choices: {str(class_desp)}
Answer Choice: {demo[1]}
"""
        qns_idx = []
        for idx, i in enumerate(test_df.iloc[start_idx:end_idx].itertuples()):
            qns_idx.append(i.Index)
            image_paths.append(os.path.join(SAVE_FOLDER, i.Index+file_suffix))
            qn_idx = idx+1
            
            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
Question {qn_idx}: What is in the image above?
Choices {qn_idx}: {str(class_desp)}

"""
        for i in range(start_idx, end_idx):
            qn_idx = i-start_idx+1
            prompt += f"""
Please respond with the following format for each question:
---BEGIN FORMAT TEMPLATE FOR QUESTION {qn_idx}---
Answer Choice {qn_idx}: [Your Answer Choice Here for Question {qn_idx}]
Confidence Score {qn_idx}: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question {qn_idx}]
---END FORMAT TEMPLATE FOR QUESTION {qn_idx}---

Do not deviate from the above format. Repeat the format template for the answer."""
        qns_id = str(qns_idx)
        for retry in range(3): 
            if (qns_id in results) and (not results[qns_id].startswith('ERROR')) and (f'END FORMAT TEMPLATE FOR QUESTION {end_idx-start_idx}' in results[qns_id]): #Skip if results exist and successful
                continue
    
            try:
                res = api(prompt, image_paths=image_paths, real_call=True, max_tokens=60*num_qns_per_round)
            except Exception as e:
                res = f'ERROR!!!! {traceback.format_exc()}'
            except KeyboardInterrupt:
                previous_usage = results.get('token_usage', (0,0,0))
                total_usage = tuple(a + b for a, b in zip(previous_usage, api.token_usage))
                results['token_usage'] = total_usage
                with open(f'{EXP_NAME}.pkl', 'wb') as f:
                    pickle.dump(results, f)
                exit()
                
            print(res)
            results[qns_id] = res

    #Update token usage and save the results
    previous_usage = results.get('token_usage', (0,0,0))
    total_usage = tuple(a + b for a, b in zip(previous_usage, api.token_usage))
    results['token_usage'] = total_usage
    with open(f'{EXP_NAME}.pkl', 'wb') as f:
        pickle.dump(results, f)
