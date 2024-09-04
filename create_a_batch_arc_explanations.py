"""
This script is used to create a batch of data in a jsonl file for calling OpenAI Batch API.
The format is:
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
...
"""

import datasets
import random
import pandas as pd
from tqdm import tqdm
import json


system_text = "You are a helpful assistant with broad expertise in general knowledge and proficiency in the Czech language."
user_text = r"""You are given a general knowledge exam question with multiple-choice answers and the correct answer provided. Generate a simple, easy-to-understand explanation in Czech. The explanation should be between 1 to 5 sentences long and written clearly in Czech. Feel free to explain step by step. Don't just rephrase the question or the answer, but provide a meaningful explanation with emphasis on reasoning.

Here are 2 simple examples of exam instances and the outputs (Vysvětlení) you need to generate:
Example 1:
Instance:
Otázka:
I když patří do stejné čeledi, orel a pelikán se liší. Jaký je mezi nimi rozdíl?
Možnosti:
A) Jejich preference pro konzumaci ryb
B) Schopnost létat
C) Způsob jejich rozmnožování
D) Způsob jejich chytání potravy
Odpověď:
D
Vysvětlení:
I když orel a pelikán patří do stejné čeledi, liší se hlavně způsobem, jakým chytají potravu. Když orel loví svou kořist, často se vrhá na svou oběť z velké výšky, zatímco pelikán používá svůj charakteristický vak pod zobákem k nabírání ryb z vody. Proto je správná odpověď - D Způsob jejich chytání potravy.

Example 2:
Instance:
Jak jsou částice v bloku železa ovlivněny, když je blok roztaven?
Možnosti:
A) Částice získávají hmotnost.
B) Částice obsahují méně energie.
C) Částice se pohybují rychleji.
D) Částice se zvětšují v objemu.
Odpověď:
C
Vysvětlení:
Když je blok železa zahřat a následně roztaven, částice v něm získávají více energie, což způsobí, že se začnou pohybovat rychleji. Toto zvýšení rychlosti pohybu částic je typické pro přechod z pevného stavu do kapalného. Proto je správná odpověď: C) Částice se pohybují rychleji.

Return only the explanation - Vysvětlení. Thus, the output must be surrounded by double quotes in format:
"Vysvětlení: 
<your explanation here>"

Here's the instance you need to generate an explaination to:
{exam_instance}
"""

dataset_path = "small_datasets/arc_challenge/arc_challenge_train.jsonl"

# create a jsonl file:
with open("batchapi/requests_arc_ch_explanations.jsonl", "w") as f:
    with open("small_datasets/arc_challenge/arc_challenge_train.jsonl", "r") as f_read:
        for i, example in enumerate(f_read):

            example_j = json.loads(example)

            # if i > 10:
                # break
            # print(i, "Number of tokens:", number_of_words*1.5)

            # Create the dictionary representing the JSON object
            json_object = {
                "custom_id": f"instance-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": user_text.format(exam_instance=example_j["combined"])}
                    ],
                    "max_tokens": 500
                }
            }

            # Convert the dictionary to a JSON string
            json_string = json.dumps(json_object)

            # Write the JSON string to the file
            f.write(json_string + '\n')
            # print(i)


# Read the file and load the list of JSON objects
# with open("batchapi/requests_arc_ch_explanations.jsonl", "r") as f:
#     for line in f:
#         json_object = json.loads(line)
#         # Process the json_object as needed
#         print(json_object["body"]["messages"][1]["content"])
#         print("\n+++++++++++++++++++\n")
#         break
    

"""from openai import OpenAI
client = OpenAI()

batch_input_file = client.files.create(
  file=open("batchinput.jsonl", "rb"),
  purpose="batch"
)"""

# Create a batch
from openai import OpenAI
client = OpenAI()

# batch_input_file = client.files.create(
#   file=open("batchapi/requests_arc_ch_explanations.jsonl", "rb"),
#   purpose="batch"
# )
# print(client.files.list())


# file_id = "file-YSC3DrNPVzfcf9cuOP8uT1Sd"
# batch = client.batches.create(
#     input_file_id=file_id,
#     endpoint="/v1/chat/completions",
#     completion_window="24h",
#     metadata={
#       "description": "nightly eval job"
#     }
# )

# print(client.batches.list())
# client.batches.cancel("batch_2TyNPmLBzP4wmXdJlDGvPALX")

# print(batch)
# Batch(id='batch_v1VlynX4Nb43JT016XWFwmMY', completion_window='24h', created_at=1723465962, endpoint='/v1/chat/completions', input_file_id='file-NUs3T6m3bxERMfMEuPsJH9Vi', object='batch', status='validating', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1723552362, failed_at=None, finalizing_at=None, in_progress_at=None, metadata={'description': 'nightly eval job'}, output_file_id=None, request_counts=BatchRequestCounts(completed=0, failed=0, total=0))


# print(client.batches.retrieve("batch_uwyyXfHWtcyX6j8GcmO90DZT"))

import json
import re


# file_response = client.files.content("file-6daSXbMLDNeacTXmOIXnJvGD")  # output_file_id in batch
# # save to a file
# with open("batchapi/arc_ch_with_explanations.jsonl", "w") as f:
#     f.write(file_response.text)



# read to a list of json objects
explanations = []
with open("batchapi/arc_ch_with_explanations.jsonl", "r") as f:
    for line in f:
        explanations.append(json.loads(line)["response"]["body"]["choices"][0]["message"]["content"])

# read to a list of json objects
initial_tasks = []
with open("small_datasets/arc_challenge/arc_challenge_train.jsonl", "r") as f:
    for line in f:
        initial_tasks.append((json.loads(line)["without_answer"], json.loads(line)["answerKey"]))

resulting_instances = []
for i in range(len(explanations)):

    formatted_instance = "<start_of_turn>user\n"+initial_tasks[i][0]+"<end_of_turn>\n<start_of_turn>model\n"+initial_tasks[i][1]+"\n"+explanations[i].strip('"')+"<end_of_turn>"

    formatted_instance = "<bos>" + formatted_instance + "<eos>"
    resulting_instances.append(formatted_instance)
    print(resulting_instances[i])
    print("---------------------------")

# save to a file
with open("small_datasets/arc_challenge/arc_challenge_train_with_explanations.jsonl", "w") as f:
    for i in range(len(resulting_instances)):
        f.write(json.dumps({"instance": resulting_instances[i]}) + "\n")


# creare a dataset object with a single column: "text"
dataset = datasets.Dataset.from_pandas(pd.DataFrame(resulting_instances, columns=["text"]))
# save to a file
dataset.save_to_disk("small_datasets/arc_challenge/arc_challenge_gemma")
