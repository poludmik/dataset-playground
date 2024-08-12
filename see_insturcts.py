import datasets
import random
from tqdm import tqdm

dataset = datasets.load_dataset('berkeley-nest/Nectar')
# dataset = datasets.load_from_disk('berkeley-nest/Nectar')

# save the dataset to disk
dataset.save_to_disk("downloads/Nectar")

# create a range of random numbers
# indices = random.sample(range(len(dataset["train"])), 10)

# randomly select 10 examples
# dataset = dataset["train"].select(indices)

# # print the dataset
# for example in dataset:
#     print(example["prompt"])
#     print("---")
#     print(example["answers"][0]["answer"])
#     print("\n\n")
    

# # Find top beginnings of all input texts through counting
# counts = {}
# for example in tqdm(dataset["train"]):
#     first_words = example["prompt"][:30]
#     if first_words in counts:
#         counts[first_words] += 1
#     else:
#         counts[first_words] = 1

# # Sort the counts
# sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

# # Print the top 10 beginnings with percentages
# print("Top 10 beginnings of input texts:")
# total = sum(counts.values())
# for i, (beginning, count) in enumerate(sorted_counts[:10]):
#     print(f"{i+1}. {beginning} ({count} examples, {count/total:.2%})")

