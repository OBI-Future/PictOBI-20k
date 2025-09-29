import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
import csv
import numpy as np
from tqdm import tqdm
import json

#-------------------------------------------------------------------------
model_path = "Qwen/Qwen2.5-VL-72B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
   model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(model_path)

with open('./quiz_data.json', 'r') as file:
    data = json.load(file)

    output_result = './qwen25-vl-72B.csv'
    count = 0

    with open(output_result, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(['query_image','correct','predicted'])
            
        for item in tqdm(data):

            query_image = item["query_image"]

            options = item["options"]

            correct_answer = item["correct_answer"]
        #-------------------------------------------------------------------------

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "./OBC_image/"+query_image,
                        }, 
                        {
                            "type": "image",
                            "image": "./object_image/"+options["A"],
                        },
                        {
                            "type": "image",
                            "image": "./object_image/"+options["B"],
                        },
                        {
                            "type": "image",
                            "image": "./object_image/"+options["C"],
                        },
                        {
                            "type": "image",
                            "image": "./object_image/"+options["D"],
                        },
                        {"type": "text", "text": "Given an pictographic character image: <image> \n \
                            and four choices: \n \
                            A: <image> \n \
                            B: <image> \n \
                            C: <image> \n \
                            D: <image> \n \
                            Please answer which real object corresponds to the meaning of the given pictographic character image? Please answer the choice (e.g., A, B, C, and D) directly without other analysis."},
                    ],
                }
            ]


            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")


            generated_ids = model.generate(**inputs, max_new_tokens=64)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
                            
            print(output_text[0])
            if 'A' in output_text[0]:
                temp = 'A'
            elif 'B' in output_text[0]:
                temp = 'B'
            elif 'C' in output_text[0]:
                temp = 'C'
            elif 'D' in output_text[0]:
                temp = 'D'

            if temp == correct_answer:
                count+=1
            
            writer.writerow([query_image,correct_answer,temp])
            
    print("--------------------------------------------------")
    print("Accuracy: ", str(count/len(data)))

    print("--------------------------------------------------")
        
            
        
    
    
    
                        
                    