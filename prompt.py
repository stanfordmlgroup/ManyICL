import traceback
import os
from tqdm import tqdm 
import random
import pickle
import numpy as np
from LMM import GPT4VAPI, GeminiAPI

def work(model, num_shot_per_class, location, num_qns_per_round, test_df, demo_df, classes, class_desp, SAVE_FOLDER, file_suffix, exclude, dataset_name, cost_analysis_rounds, detail='auto'):
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    EXP_NAME = f'{dataset_name}_{num_shot_per_class*len(classes)}shot_{model}_{num_qns_per_round}'
    
    if model.startswith('gpt'):
        api = GPT4VAPI(model=model, detail=detail)
        exclude_list = exclude['GPT']
    else:
        assert model=='Gemini1.5'
        api = GeminiAPI(location=location)
        exclude_list = exclude['Gemini']
    print(EXP_NAME, f'test size = {len(test_df)}')
    
    
    demo_examples = []
    for class_name in classes:
        num_cases_class = 0
        for j in demo_df[demo_df[class_name] == 1].itertuples():
            if num_cases_class == num_shot_per_class:
                break
            if j.Index in exclude_list:
                continue
            demo_examples.append((j.Index, class_desp[class_to_idx[class_name]]))
            num_cases_class += 1
    
    assert len(demo_examples) == num_shot_per_class*len(classes)
    
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
    previous_usage = results.get('token_usage', (0,0,0))
    total_usage = tuple(a + b for a, b in zip(previous_usage, api.token_usage))
    results['token_usage'] = total_usage
    with open(f'{EXP_NAME}.pkl', 'wb') as f:
        pickle.dump(results, f)
