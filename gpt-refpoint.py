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
    base_url = "xxxxxxxxxxxxx",
    api_key = "xxxxxxxxxxxxxx"
)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


model_name = "gpt-4o-2024-11-20" 

#-------------------------------------------------------------------------
with open('./quiz_reference_point.json', 'r') as file:
    data = json.load(file)



    output_result = './gpt4o-refpoint.csv'
    count = 0

    with open(output_result, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(['OBC','object','correct','predicted'])
            
        for item in tqdm(data):
            
            subfolder = item["subfolder"]
            
            OBC_image = item["OBC_image"]
           
            object_image = item["object_image"]
            answer = item["answer"]
          
        #-------------------------------------------------------------------------
        
            base64_image1 = encode_image( "./quiz_reference_point/"+subfolder+'/'+OBC_image[:-4]+'_REF1_annotated.png')
            base64_image2 = encode_image("./quiz_reference_point/"+subfolder+'/'+object_image[:-4]+'_final_annotated.png')
            
                
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
                            "type": "text",
                            "text": "Given an Oracle bone character image <image> and a real object image <image> \n \
                            Please answer which point best corresponds to the REF1 point? Answer the choice (e.g., A, B, C, and D) directly without other analysis."
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
                
            if temp == answer:
                count+=1
            
            writer.writerow([OBC_image,object_image,answer,temp])
            
    print("--------------------------------------------------")
    print("Accuracy: ", str(count/len(data)))

    print("--------------------------------------------------")
                



