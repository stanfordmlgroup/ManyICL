import base64
import time
import pickle
import os
import uuid
import pandas as pd
from tqdm import tqdm 
import traceback
import random
from PIL import Image

from openai import OpenAI

import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models


class GPT4VAPI():
    def __init__(self, model='gpt-4-turbo-2024-04-09', img_token = '<<IMG>>', TPM = 10000, RPD = 500, seed=66, temperature=0, stats_loc='api_usage.pkl', save_folder = 'GPT4V_response/', detail='low'):
        self.model = model
        self.img_token = img_token
        self.TPM = TPM
        self.seed = seed
        self.temperature = temperature
        self.save_folder = save_folder
        self.detail = detail
        os.makedirs(save_folder, exist_ok=True)
        self.client = OpenAI(api_key="YOUR_OPENAI_API_KEY_HERE")
        if os.path.isfile(stats_loc):
            with open(stats_loc, 'rb') as f:
                self.token_usage = pickle.load(f)
        else:
            self.token_usage = (0, 0, 0) #completion_tokens=38, prompt_tokens=195, total_tokens=233
        self.response_times = []
        
    def generate_image_url(self, image_path, detail='low'):
        # Given an image_path, return a dict
        # Function to encode the image
        def encode_image(image_path):
          if str(image_path).lower().endswith('tif'):
            with Image.open(image_path) as img:
              img.convert('RGB').save('temp.jpeg', 'JPEG')
            image_path = 'temp.jpeg'
          with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        return {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64, {encode_image(image_path)}",
            "detail": detail
          }
        }
    
    def generate_text_url(self, text):
        return {
            "type": "text",
            "text": text
        }
    
    def __call__(self, prompt, image_paths = [], real_call=True, count_time=False, max_tokens = 50, content_only = True):
        if not isinstance(image_paths, list): #For single file
            image_paths = [image_paths]
        prompt = prompt.split(self.img_token)
        assert (len(prompt) == len(image_paths) + 1)
        if prompt[0] != '':
            messages = [self.generate_text_url(prompt[0])]
        else:
            messages = []
        for idx in range(1, len(prompt)):
            messages.append(self.generate_image_url(image_paths[idx - 1], detail=self.detail))
            if prompt[idx].strip() != '':
                messages.append(self.generate_text_url(prompt[idx]))
        if not real_call:
            return messages
        start_time = time.time()
        response = self.client.chat.completions.create(
            model = self.model,
            messages = [{
                "role": "user",
                "content": messages
            }],
            max_tokens=min(4096, max_tokens),
            temperature=self.temperature,
            seed=self.seed
        )
        
        end_time = time.time()
        self.response_times.append(end_time - start_time)

        results = [prompt, image_paths, response, end_time-start_time]
        with open(os.path.join(self.save_folder, str(uuid.uuid1())+'.pkl'), 'wb') as f:
            pickle.dump(results, f)

        try:
            self.token_usage = (self.token_usage[0]+response.usage.completion_tokens, self.token_usage[1]+response.usage.prompt_tokens, self.token_usage[2]+response.usage.total_tokens)
        except:
            pass
        if content_only:
            return response.choices[0].message.content
        else:
            return response

class GeminiAPI():
    def __init__(self, model="gemini-1.5-pro-preview-0409", img_token = '<<IMG>>', TPM = 10000, RPM=5, RPD = 500, seed=66, temperature=0, stats_loc='api_usage.pkl', save_folder = 'Gemini1.5_response/', location="us-central1"):
        self.model = model
        self.img_token = img_token
        self.TPM = TPM
        self.seed = seed
        self.temperature = temperature
        self.save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)
        vertexai.init(project="YOUR_GCP_PROJECT_ID_HERE", location=location)
        self.client = GenerativeModel(model)

        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        if os.path.isfile(stats_loc):
            with open(stats_loc, 'rb') as f:
                self.token_usage = pickle.load(f)
        else:
            self.token_usage = (0, 0, 0) #completion_tokens=38, prompt_tokens=195, total_tokens=233
        
        self.response_times = []
        self.last_time = None
        self.interval = 0.5+60/ RPM
        
    def generate_image_url(self, image_path):
        # Given an image_path, return a dict
        # Function to encode the image
        def encode_image(image_path):
          with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        image1 = Part.from_data(mime_type="image/jpeg",data=encode_image(image_path))
        return image1
    
    def __call__(self, prompt, image_paths = [], real_call=True, max_tokens = 50, content_only = True):
        if self.last_time is not None:
            # Calculate how much time the loop took
            end_time = time.time()
            elapsed_time = end_time - self.last_time
            # Wait for the remainder of the interval, if necessary
            if elapsed_time < self.interval:
                time.sleep(self.interval - elapsed_time)
        
        if not isinstance(image_paths, list): #For single file
            image_paths = [image_paths]
        prompt = prompt.split(self.img_token)
        assert (len(prompt) == len(image_paths) + 1)
        if prompt[0] != '':
            messages = [prompt[0]]
        else:
            messages = []
        for idx in range(1, len(prompt)):
            messages.append(self.generate_image_url(image_paths[idx - 1]))
            if prompt[idx].strip() != '':
                messages.append(prompt[idx])
        if not real_call:
            return messages
        
        start_time = time.time()
        self.last_time = start_time
        responses = self.client.generate_content(
          messages,
          generation_config={"max_output_tokens": min(max_tokens, 8192), "temperature": self.temperature},
          safety_settings=self.safety_settings,
          stream=False,
        )
        
        end_time = time.time()
        self.response_times.append(end_time - start_time)
        
        results = [prompt, image_paths, responses, end_time-start_time]
        with open(os.path.join(self.save_folder, str(uuid.uuid1())+'.pkl'), 'wb') as f:
            pickle.dump(results, f)

        try:
            usage = responses._raw_response.usage_metadata
            self.token_usage = (self.token_usage[0]+usage.candidates_token_count, self.token_usage[1]+usage.prompt_token_count, self.token_usage[2]+usage.total_token_count)
        except:
            pass
        if content_only:
            return responses.text
        else:
            return responses
