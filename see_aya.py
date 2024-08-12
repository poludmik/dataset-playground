import datasets
import random
from tqdm import tqdm

# Load the dataset from the folder "downloads/aya_hotpotqa_ces":
# dataset = datasets.load_from_disk('datasets/downloads/aya_all_czech_google_gemma-2-2b-it')
dataset = datasets.load_from_disk('datasets/downloads/aya_all_czech')

# create a range of random numbers
# indices = random.sample(range(len(dataset["train"])), 20)

# randomly select 10 examples
# dataset = dataset["train"].select(indices)

# # print the dataset
# for example in dataset:
#     print(example["input"])
#     print("---")
#     print(example["output"])
#     print("\n\n")
    

# Find top beginnings of all input texts through counting
counts = {}
for example in tqdm(dataset["train"]):
    first_words = example["inputs"][:30]
    if first_words in counts:
        counts[first_words] += 1
    else:
        counts[first_words] = 1

# Sort the counts
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

# Print the top 10 beginnings with percentages
print("Top 10 beginnings of input texts:")
total = sum(counts.values())
for i, (beginning, count) in enumerate(sorted_counts[:10]):
    print(f"{i+1}. {beginning} ({count} examples, {count/total:.2%})")

