import datasets
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm


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
                return f"<bos><start_of_turn>user\n{instance_in}<end_of_turn>\n<start_of_turn>model\n", f"{instance_out}<end_of_turn>\n"
            
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str, help="Dataset folder")
    parser.add_argument("--m", type=str, help="Model name")
    parser.add_argument("--use_tokenizer", type=bool, default=False, help="Use tokenizer")
    args = parser.parse_args()

    apply_format_and_save(args.d, args.m, use_tokenizer=args.use_tokenizer)
