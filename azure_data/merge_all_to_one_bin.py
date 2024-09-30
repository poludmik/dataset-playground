"""
Given a list of zst files this script will merge all the files into one binary file like in the example above.
"""

import math
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import zstandard as zstd
import json
import tqdm

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
# model = AutoModelForCausalLM.from_pretrained('gpt2')

# List of files to process
list_of_files_or_folders = [
    "czech_llm_data/czech-llm-dataset-complete/cswiki/20231101/raw/cswiki.jsonl.zst",
    "czech_llm_data/czech-llm-dataset-complete/czech-socio-review/raw/czech-socio-review.jsonl.zst",
    "czech_llm_data/czech-llm-dataset-complete/idnes/raw/idnes.jsonl.zst",
    "czech_llm_data/czech-llm-dataset-complete/mlp-books/raw/mlp-books.jsonl.zst",
    "czech_llm_data/czech-llm-dataset-complete/patents/raw/patents.jsonl.zst",
    "czech_llm_data/czech-llm-dataset-complete/plenary-speeches/plenary-speeches.jsonl.zst",
    "czech_llm_data/czech-llm-dataset-complete/syn/v9/raw/syn_v9.jsonl.zst",
    "czech_llm_data/czech-llm-dataset-complete/theses/raw/theses.jsonl.zst",
    "czech_llm_data/czech-llm-dataset-complete/tinystories/tinystories_cs_train.jsonl.zst"
]

# Define the output binary file
output_file = "czech_llm_data/czech-llm-dataset-complete/merged_all_files.bin"
max_length = 1024
stride = 1024
pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id  # Use pad_token or EOS if padding is undefined
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id

# Function to process and tokenize data and write it into a binary file
def tokenize_and_save_zst_files(list_of_files, output_file, tokenizer, max_length=1024, stride=1024, max_lines=3):
    # Open binary file for writing
    with open(output_file, 'wb') as bin_file:
        for file in list_of_files:
            print(f"Processing file here: {file}")
            with open(file, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    print(f"Processing file: {file}")
                    buffer = b""
                    line_count = 0  # Counter to limit lines processed
                    while True:
                        chunk = reader.read(8192)
                        if not chunk:
                            break
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            obj = json.loads(line)
                            text = obj.get("text", "")

                            # TODO for gemma
                            # text = "<bos>" + text + "<eos>"

                            # Tokenize the text and add <bos> and <eos> tokens
                            encodings = tokenizer.encode(text, add_special_tokens=True)

                            # Add <bos> and <eos> tokens
                            encodings = [bos_token_id] + encodings # + [eos_token_id]
                            
                            # Chunking the tokenized sequence
                            for i in range(0, len(encodings), stride):
                                chunk = encodings[i:i + max_length]

                                # Pad the chunk if it is shorter than max_length
                                # if len(chunk) < max_length:
                                #     chunk += [pad_token_id] * (max_length - len(chunk))

                                # Write chunk to the binary file as numpy uint32 array
                                arr = np.array(chunk, dtype=np.uint32)
                                arr.tofile(bin_file)

                            line_count += 1
                            if line_count % 10 == 0:
                                print(f"Processed {line_count} lines from {file}")
                            # if line_count >= max_lines:
                            #     print(f"Processed {max_lines} lines from {file}")
                            #     break  # Stop after processing 3 lines
                        # if line_count >= max_lines:
                        #     break

    print(f"Tokenized data saved in {output_file}")

# Tokenize and save the dataset (process only 3 lines for debugging)
tokenize_and_save_zst_files(list_of_files_or_folders, output_file, tokenizer, max_length, stride, max_lines=3)

# Check the size of the output binary file
print(f"Total size of the binary file: {Path(output_file).stat().st_size / (1024 * 1024):.2f} MB")


####################################################################################################


# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import math

# # Custom Dataset class for reading from the binary file
# class TokenizedDataset(Dataset):
#     def __init__(self, data_file, block_size):
#         self.data_file = data_file
#         self.block_size = block_size
#         # Memory-map the file for efficient reading
#         self.data = np.memmap(self.data_file, dtype=np.uint32, mode='r')
#         self.data_len = math.ceil(len(self.data) / self.block_size)
    
#     def __len__(self):
#         return self.data_len
    
#     def __getitem__(self, idx):
#         # Extract a block (chunk) of tokens
#         start_idx = idx * self.block_size
#         end_idx = start_idx + self.block_size
        
#         # Handle the case where the last chunk might be shorter than block_size
#         x = torch.from_numpy(self.data[start_idx:end_idx].astype(np.uint32))
        
#         # Shift the input to create labels (next token prediction)
#         y = torch.from_numpy(self.data[start_idx+1:end_idx+1].astype(np.uint32))
        
#         # If the last chunk is shorter than block_size, pad with 0
#         if x.size(0) < self.block_size:
#             pad_size = self.block_size - x.size(0)
#             pad_tensor = torch.zeros(pad_size, dtype=torch.uint32)
#             x = torch.cat((x, pad_tensor), dim=0)
#             y = torch.cat((y, pad_tensor), dim=0)
        
#         return {'input_ids': x, 'labels': y}

# # Define the file path to the binary file and other parameters
# data_file = 'czech_llm_data/czech-llm-dataset-complete/merged_all_files.bin'  # Binary file path
# block_size = 1024  # Model's context length
# batch_size = 1  # Number of samples per batch (adjust based on memory)

# # Initialize the dataset and dataloader
# dataset = TokenizedDataset(data_file, block_size)
# dataloader = DataLoader(
#     dataset,
#     batch_size=batch_size,
#     # num_workers=4,  # Adjust the number of workers for faster loading
#     shuffle=False  # Enable shuffling for training
# )

# # Iterate through the dataloader
# for batch in dataloader:
#     input_ids = batch['input_ids']
#     labels = batch['labels']
    
#     # Example: print the first input and label of the batch
#     print()
#     print("Input IDs:", input_ids[0], input_ids[0].shape[0])
#     print("Labels:", labels[0], labels[0].shape[0])

#     print("Input Text:", tokenizer.decode(input_ids[0], skip_special_tokens=False))
#     print("Label Text:", tokenizer.decode(labels[0], skip_special_tokens=False))
#     # break  # Remove or comment this break in actual training loop
#     input()
