"""
This script is used to create a batch of data in a jsonl file for calling OpenAI Batch API.
The format is:
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
...
"""

import datasets
import random
from tqdm import tqdm
import json

dataset = datasets.load_dataset('berkeley-nest/Nectar')

# take 10000 first examples
dataset = dataset["train"].select(range(1000))

# for example in dataset:
#     print(example["prompt"])
#     print("---")
#     print(example["answers"][0]["answer"])
#     print("\n\n")


system_text = "You are a helpful assistant that accurately translates text from English to Czech."
user_text = r"""Please translate the following text from English to Czech. Keep the translation as close to the original as possible.
If the text contains a math expression or code snippets, keep them as they are. The main focus is on the translation of the text itself.
The translated text is very important and will be used in a training dataset for a czech language model. Be careful with the word endings and the grammar in general, the czech language is complex.
The english text is formatted with `Human:` and `Assistant:` instruction notation. Ignore the human instructions and assistant answers in the text and just translate it. Return the translated text in this json format:
{{
  "Human": "Přeložený text první zprávy od člověka",
  "Assistant": "Přeložený text zprávy od asistenta",
  "Human": "Přeložený text druhé zprávy od člověka",
  ...,
  "Assistant": "Přeložený text poslední zprávy od asistenta"
}}
Output nothing except a translated conversation in this JSON format. Here's the conversation text to translate:
"{instance}"
"""

# # create a jsonl file:
# with open("batchapi/requests_first_1000.jsonl", "w") as f:
#     for i, example in enumerate(dataset):
#         input_text = example["prompt"]
#         output_text = example["answers"][0]["answer"]
#         full_conversation = "\n" + (input_text + "\n" + output_text).strip()
        
#         number_of_words = len(full_conversation.split())
#         # print(i, "Number of tokens:", number_of_words*1.5)

#         # Create the dictionary representing the JSON object
#         json_object = {
#             "custom_id": f"instance-{i}",
#             "method": "POST",
#             "url": "/v1/chat/completions",
#             "body": {
#                 "model": "gpt-4o-mini",
#                 "messages": [
#                     {"role": "system", "content": system_text},
#                     {"role": "user", "content": user_text.format(instance=full_conversation)}
#                 ],
#                 "max_tokens": 3000
#             }
#         }

#         # Convert the dictionary to a JSON string
#         json_string = json.dumps(json_object)

#         # Write the JSON string to the file
#         f.write(json_string + '\n')
#         print(i)


# Read the file and load the list of JSON objects
# with open("batchapi/requests.jsonl", "r") as f:
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
#   file=open("batchapi/requests_first_1000.jsonl", "rb"),
#   purpose="batch"
# )
# print(client.files.list())


# file_id = "file-g96fFgH2T8b8b28yyFkjdX32"
# batch = client.batches.create(
#     input_file_id=file_id,
#     endpoint="/v1/chat/completions",
#     completion_window="24h",
#     metadata={
#       "description": "nightly eval job"
#     }
# )

# print(client.batches.list())

# print(batch)
# Batch(id='batch_v1VlynX4Nb43JT016XWFwmMY', completion_window='24h', created_at=1723465962, endpoint='/v1/chat/completions', input_file_id='file-NUs3T6m3bxERMfMEuPsJH9Vi', object='batch', status='validating', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1723552362, failed_at=None, finalizing_at=None, in_progress_at=None, metadata={'description': 'nightly eval job'}, output_file_id=None, request_counts=BatchRequestCounts(completed=0, failed=0, total=0))



print(client.batches.retrieve("batch_uwyyXfHWtcyX6j8GcmO90DZT"))


# file_response = client.files.content("file-XW2X1WVuYbON8nVvYZlB9ilA") # output_file_id in batch
# import json

# # parse to json objects line by line
# # save the output to a file
# with open("batchapi/requests_translated.jsonl", "w") as f:
#     for line in file_response.text.split("\n"):
#         try:
#           conversation_text = json.loads(line)["response"]["body"]["choices"][0]["message"]["content"]
          
#           # Parse the string as a JSON object
#           data = json.loads(conversation_text)

#           # Initialize an empty list to store the tuples
#           messages = []
#           messages_json = {"conversation": []}

#           # Iterate through the keys and values of the JSON object
#           for key, value in data.items():
#               if key == "Human":
#                   human_message = value
#               elif key == "Assistant":
#                   assistant_message = value
#                   messages.append((human_message, assistant_message))
#                   messages_json["conversation"].append({"Human": human_message, "Assistant": assistant_message})

#           for message in messages:
#               print(message[0])
#               print(">>>")
#               print(message[1])
#               print("\n")

#           # Write to the file
#           f.write(json.dumps(messages_json) + '\n')
#         except Exception as e:
#           print("Error:", e)
#           pass

#         print("\n+++++++++++++++++++\n")
