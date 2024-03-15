import os
import time
import base64
import requests

from openai import OpenAI, BadRequestError
from model.base import LargeMultimodalModel

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class GPTClient(LargeMultimodalModel):
    def __init__(self):
        pass
    
    def forward(self, image_paths, prompt):
        client = OpenAI(api_key="XXXXXX")

        # Getting the base64 string
        base64_images = [encode_image(p) for p in image_paths]
        
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    prompt,
                    *map(lambda x: {"image": x}, base64_images),
                ],
            },
        ]
        params = {
            "model": "gpt-4-vision-preview",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 4096,
        }

        while 1:
            try:
                result = client.chat.completions.create(**params)
                response = result.choices[0].message.content
                return response
            except Exception as e:
                if isinstance(e, BadRequestError):
                    return "Yes"
                print(e, flush=True)
                print('Timeout error, retrying...')
                time.sleep(5)
