import datasets
import os
import json

dataset = datasets.load_dataset("CIIRC-NLP/arc-cs", "ARC-Challenge", split="train")
print(dataset)

i = 0
print(dataset[i]["question"])
choices = []
# "choice" contains "text" array and "label" array with 5 elements
# print(dataset[i]["choices"]["text"])
# print(dataset[i]["choices"]["label"])

# zip text and label
for text, label in zip(dataset[i]["choices"]["text"], dataset[i]["choices"]["label"]):
    choices.append({"text": text, "label": label + ") "})

# convert to string
choices = "\n".join([f"{choice['label']}{choice['text']}" for choice in choices])
print(choices)

# answerKey
print(dataset[i]["answerKey"])

# Process all data and save to a jsonl file:
with open("small_datasets/arc_challenge/arc_challenge_train.jsonl", "w") as f:
    for i, example in enumerate(dataset):
        print(f"\rProcessing example {i+1} / {len(dataset)}", end="")
        question = example["question"]
        choices = []
        for text, label in zip(example["choices"]["text"], example["choices"]["label"]):
            choices.append({"text": text, "label": label + ") "})
        choices = "\n".join([f"{choice['label']}{choice['text']}" for choice in choices])
        answerKey = example["answerKey"]
        
        combined = "Otázka:\n" + question + "\nMožnosti:\n" + choices + "\nOdpověď:\n" + answerKey
        without_answer = "Otázka:\n" + question + "\nMožnosti:\n" + choices + "\nOdpověď:\n"
        # Create a dictionary for the JSON line
        json_line = {
            "question": question,
            "choices": choices,
            "answerKey": answerKey,
            "combined": combined,
            "without_answer": without_answer
        }

        # Write the JSON line to the file
        f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

        print(combined)
        print(">>>>>>>>>>>>>>>>>>>>")