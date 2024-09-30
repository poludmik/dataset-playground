import json
import zstandard as zstd

# file_path = "czech_llm_data/czech-llm-dataset-complete/commoncrawl/august-2022/cleaned-deduplicated-url_deduplicated/august-2022-url_deduplicated.jsonl.zst"
# file_path = "czech_llm_data/czech-llm-dataset-complete/cswiki/20231101/raw/cswiki.jsonl.zst"
# file_path = "czech_llm_data/czech-llm-dataset-complete/culturax/url_deduplicated/3-url_deduplicated.jsonl.zst" # didn't open
# file_path = "czech_llm_data/czech-llm-dataset-complete/czech-socio-review/raw/czech-socio-review.jsonl.zst"
# file_path = "czech_llm_data/czech-llm-dataset-complete/hplt/v1.2/url_deduplicated/1-url_deduplicated.jsonl.zst"
# file_path = "czech_llm_data/czech-llm-dataset-complete/idnes/raw/idnes.jsonl.zst"
file_path = "czech_llm_data/czech-llm-dataset-complete/mlp-books/raw/mlp-books.jsonl.zst"
# file_path = "czech_llm_data/czech-llm-dataset-complete/patents/raw/patents.jsonl.zst"
# file_path = "czech_llm_data/czech-llm-dataset-complete/plenary-speeches/plenary-speeches.jsonl.zst
# file_path = "czech_llm_data/czech-llm-dataset-complete/syn/v9/raw/syn_v9.jsonl.zst"
# file_path = "czech_llm_data/czech-llm-dataset-complete/theses/raw/theses.jsonl.zst"
#file_path = "czech_llm_data/czech-llm-dataset-complete/tinystories/tinystories_cs_train.jsonl.zst"

def read_zst_file(file_path):
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            buffer = b""
            while True:
                chunk = reader.read(8192)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    yield json.loads(line)

output_file = "czech_llm_data/czech-llm-dataset-complete/mlp-books/raw/mlp-books-filtered.jsonl"
# clear output file if it exists
open(output_file, 'w').close()

for i, record in enumerate(read_zst_file(file_path)):
    # if i > 5:
    #     break
    text = record['text']
    lines = text.split("\n")
    keep = ""
    for line in lines:
        if len(line) > 500:
            keep += line + "\n"
    record['text'] = keep
    with open(output_file, 'a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
