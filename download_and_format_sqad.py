import datasets

# Load the dataset from Pehy/cs_sqad-3.0 dataset and train split:
dataset = datasets.load_dataset("Pehy/cs_sqad-3.0", split="train")

# For each example in the dataset, take the context, the question and the answer and format them together:
"""
Kontext:
... "jmenovat", ale je současně i volbou jeho osobního patrona, průvodce a životního vzoru tohoto jména. Proto, když církev slaví památku daného svatého, slaví jej i ti, kdo nesou jeho jméno, neboť je to den jejich patrona. Bývá zvykem dávat při křtu i jmen víc. V některých katolických zemích se přibírá ještě další jméno i při biřmování, např. v Česku. V Itálii naopak tento zvyk neznají. Při oslavách je zvykem nositelům jména popřát a případně je i obdarovat. Vzhledem k množství světců je mnoho rodných jmen příslušných k více dnům a naopak, k jednotlivým dnům připadá řada svatých. Ve Švédsku publikuje oficiální seznam jmenin Královská švédská akademie věd. Český občanský kalendář původně sice vycházel z církevního kalendáře, ale v průběhu času doznal mnoha změn...
Otázka:
Kdo ve Švédsku publikuje oficiální seznam jmenin?
Odpověď:
Královská švédská akademie věd
"""

formatted_examples = []
for example in dataset:
    formatted_example = f"""<bos><start_of_turn>user\nKontext:\n{example["context"]}\nOtázka:\n{example["question"]}\nOdpověď:\n<end_of_turn>\n<start_of_turn>model\n{example["answers"]["text"][0]}<end_of_turn><eos>"""
    formatted_examples.append(formatted_example)
    # print(formatted_example)

# Save the formatted examples to a dataset locally as a dataset object to "small_datasets/sqad_formatted":
dataset = datasets.Dataset.from_dict({"text": formatted_examples})

# Save the dataset to a file:
dataset.save_to_disk("small_datasets/sqad_formatted")


# read the dataset from the file:
dataset = datasets.load_from_disk("small_datasets/sqad_formatted")
print(dataset[0]["text"])
