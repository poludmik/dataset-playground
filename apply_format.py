import datasets
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import re


def format_one_instance(instance_in, instance_out, model_name, use_tokenizer=False):
    try:
        if use_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            message_user = [
                {"role": "user", "content": instance_in}
            ]
            messages_full = [
                {"role": "user", "content": instance_in},
                {"role": "model", "content": instance_out}
            ]
            input_text = tokenizer.apply_chat_template(message_user, 
                                              tokenize=False,
                                              add_generation_prompt=True)
            all_text = tokenizer.apply_chat_template(messages_full, 
                                              tokenize=False,
                                              add_generation_prompt=False)

            # now need to remove the input_text from all_text's beginning
            out_text = all_text[len(input_text):]
            return input_text, out_text
        else:
            if model_name == "google/gemma-2-2b-it":
                return f"<bos><start_of_turn>user\n{instance_in}<end_of_turn>\n<start_of_turn>model\n", f"{instance_out}<end_of_turn><eos>"
            
    except Exception as e:
        print(f"Error loading tokenizer(formatter) for model {model_name}: {e}")
        return None


def apply_format_and_save(dataset_folder, model_name, use_tokenizer=False):
    dataset = datasets.load_from_disk(dataset_folder)
    
    # Initialize the final dataset
    final_dataset = datasets.DatasetDict()

    for key in dataset.keys():
        print(f"Processing {key}...")
        input_list = []
        output_list = []
        source_list = []

        for i, instance in tqdm(enumerate(dataset[key])):
            input_text, out_text = format_one_instance(instance["inputs"], instance["targets"], model_name, use_tokenizer=use_tokenizer)
            if input_text and out_text:
                input_list.append(input_text)
                output_list.append(out_text)
                source_list.append(dataset_folder)
            
            # Save in batches to avoid memory issues
            if i % 1000 == 0 and i > 0:
                temp_dataset = datasets.Dataset.from_dict({"input": input_list, "output": output_list, "source": source_list})
                if key in final_dataset:
                    final_dataset[key] = datasets.concatenate_datasets([final_dataset[key], temp_dataset])
                else:
                    final_dataset[key] = temp_dataset
                input_list = []
                output_list = []
                source_list = []

        # Save remaining data
        if input_list:
            temp_dataset = datasets.Dataset.from_dict({"input": input_list, "output": output_list, "source": source_list})
            if key in final_dataset:
                final_dataset[key] = datasets.concatenate_datasets([final_dataset[key], temp_dataset])
            else:
                final_dataset[key] = temp_dataset

    model_name = model_name.replace("/", "_")
    dataset_folder = dataset_folder.rstrip("/")
    final_dataset.save_to_disk(f"{dataset_folder}_{model_name}")

def unescape_string(s):
    s = s.replace(r'\n', '\n')
    s = s.replace(r'\t', '\t')
    s = s.replace(r'\"', '"')
    s = s.replace(r'\\', '\\')
    # Add more replacements if necessary
    return s

def format_one_instance_multiturn(conversation_list_of_pairs, model_name, use_tokenizer=False):
    try:
        if model_name == "google/gemma-2-2b-it":
            current_input_text = ""
            current_output_text = ""

            for pair in conversation_list_of_pairs:
                # Usage remains the same
                human_message = unescape_string(pair['Human'])
                assistant_message = unescape_string(pair['Assistant'])
                
                new_human_input = f"<start_of_turn>user\n{human_message}<end_of_turn>\n<start_of_turn>model\n"
                new_output = f"{assistant_message}<end_of_turn>\n"

                current_input_text += current_output_text + new_human_input
                current_output_text = new_output

                yield "<bos>" + current_input_text, current_output_text + "<eos>"

    except Exception as e:
        print(f"Error formatting for {model_name}: {e}")
        return None


def apply_format_on_multiturn_jsonl(jsonl_file, model_name, use_tokenizer=False):
    with open(jsonl_file, "r") as f:
        data = f.readlines()
        # Each line holds a conversation instance in a format like:
        # {"instance_id": "instance-0", "conversation": [{"Human": "Hello", "Assistant": "Hi"}, {"Human": "How are you?", "Assistant": "I'm fine."}]}
        # When the conversation is longer than n qa pairs, the dataset will contain n instances, which are incremental
        # For the previous example, the dataset will contain 2 instances:
        # f"<bos><start_of_turn>user\n{human_message_1}<end_of_turn>\n<start_of_turn>model\n", f"{assistant_message_1}<end_of_turn>\n"
        # f"<bos><start_of_turn>user\n{human_message_1}<end_of_turn>\n<start_of_turn>model\n"{assistant_message_1}<end_of_turn>\n<start_of_turn>user\n{human_message_2}<end_of_turn>\n<start_of_turn>model\n", f"{assistant_message_2}<end_of_turn>\n"

        # Initialize the final dataset as a dictionary of lists
        dataset_entries = {
            "input": [],
            "output": [],
            "source": []
        }

        # iterate over the lines - instances
        for i, line in tqdm(enumerate(data)):
            # load the instance as a json object
            instance = json.loads(line)
            source = instance["instance_id"] # e.g. "instance-0"
            conversation = instance["conversation"] # a list with qa pairs

            for input_text, out_text in format_one_instance_multiturn(conversation, model_name, use_tokenizer=use_tokenizer):
                # Add the input, output, and source to the dataset entries
                
                print(input_text)
                print("----")
                print(out_text)
                print("++++++++++++++++\n")

                if input_text and out_text:
                    dataset_entries["input"].append(input_text)
                    dataset_entries["output"].append(out_text)
                    dataset_entries["source"].append(source)
                
        # Save the dataset entries to a dataset
        dataset = datasets.Dataset.from_dict(dataset_entries)
        model_name = model_name.replace("/", "_")

        # remove the .jsonl extension and the folder path
        jsonl_file = jsonl_file.rstrip(".jsonl").split("/")[-1]
        dataset.save_to_disk(f"downloads/{jsonl_file}_{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, help="Dataset folder")
    parser.add_argument("--m", type=str, help="Model name")
    parser.add_argument("--use_tokenizer", type=bool, default=False, help="Use tokenizer")
    parser.add_argument("--from_batchapi", type=bool, default=False, help="Data from batchapi")
    args = parser.parse_args()

    if args.from_batchapi:
        apply_format_on_multiturn_jsonl(args.d, args.m, use_tokenizer=args.use_tokenizer)
    else:
        apply_format_and_save(args.d, args.m, use_tokenizer=args.use_tokenizer)
