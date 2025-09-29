import base64
import requests
from tqdm import tqdm
import time
import csv
import re
import os
from PIL import Image
from openai import OpenAI
import numpy as np
import subprocess

import json

client = OpenAI(
    base_url = "http://35.220.164.252:3888/v1/",
    api_key = "sk-wc3xglLgyAcbBEHRagx6WSCZ9LksRFB50UdN5URoobqHjRXK"
)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


model_name = "gpt-4o-2024-11-20"  


with open('./quiz_data.json', 'r') as file:
    data = json.load(file)

    output_result = './results/'+model_name+'.csv'
    count = 0

    with open(output_result, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(['query_image','correct','predicted'])

        for item in tqdm(data):

            query_image = item["query_image"]

            options = item["options"]

            correct_answer = item["correct_answer"]
            
            temp = ""
        #-------------------------------------------------------------------------
        
            base64_image1 = encode_image("./OBC_image/"+query_image)
            base64_image2 = encode_image("./object_image/"+options["A"])
            base64_image3 = encode_image("./object_image/"+options["B"])
            base64_image4 = encode_image("./object_image/"+options["C"])
            base64_image5 = encode_image("./object_image/"+options["D"])
                
            response = client.chat.completions.create(
            model=model_name,  
            messages=[ 
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image1}"
                            },
                        },
                            {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image2}"
                            },
                        },
                            {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image3}"
                            },
                        },
                            {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image4}"
                            },
                        },
                            {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image5}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Given an Oracle bone character image: <image> \n \
                            and four choices: \n \
                            A: <image> \n \
                            B: <image> \n \
                            C: <image> \n \
                            D: <image> \n \
                            Please answer which real object corresponds to the meaning of the given Oracle bone character image? Please answer the choice (e.g., A, B, C, and D) directly without other analysis."
                        }
                        ]
                    }
                    ],
                    max_tokens = 64,
            )
            
            
            content = response.choices[0].message.content
            
            print(content)
            if 'A' in content:
                temp = 'A'
            elif 'B' in content:
                temp = 'B'
            elif 'C' in content:
                temp = 'C'
            elif 'D' in content:
                temp = 'D'
            else:
                temp = None
                
            if temp == correct_answer:
                count+=1
            
            writer.writerow([query_image,correct_answer,temp])
            
    print("--------------------------------------------------")
    print("Accuracy: ", str(count/len(data)))

    print("--------------------------------------------------")
                



