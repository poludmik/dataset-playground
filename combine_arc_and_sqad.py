import datasets
import pandas as pd


# read the arc dataset from local folder
arc = datasets.load_from_disk('small_datasets/arc_challenge/arc_challenge_gemma/')

# read the sqad dataset from local folder
sqad = datasets.load_from_disk('small_datasets/sqad_formatted/')

print(arc)
print(sqad)

# shuffle both datasets
seed = 228
# arc = arc.shuffle(seed=seed)
sqad = sqad.shuffle(seed=seed)

# select 500 from each dataset
# arc = arc.select(range(500))
sqad = sqad.select(range(500))

new_arc = []
# for each "text" in the arc dataset, remove the text inside that is between the "Vysvětlení:" and "<eos>"
for i in range(len(arc)):
    text = arc[i]["text"]
    text_start = text.split("\nVysvětlení:")[0]
    # print(text_start)
    # arc[i]["text"] = text_start + "<end_of_turn><eos>"
    new_arc.append(text_start+ "<end_of_turn><eos>")

arc = datasets.Dataset.from_pandas(pd.DataFrame(new_arc, columns=["text"]))

# combine the two datasets
# combined = datasets.concatenate_datasets([arc, sqad])
combined = arc
# shuffle the combined dataset
# combined = combined.shuffle(seed=seed)
print(combined)

# print(combined[0]["text"])
for i in range(100):
    print(combined[i]["text"])
    print("")

# save the combined dataset to local folder
combined.save_to_disk('small_datasets/arc_no_expl_gemma/')