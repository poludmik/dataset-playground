import json
import zstandard as zstd

file_path = "czech_llm_data/czech-llm-dataset-complete/syn/v9/raw/syn_v9.jsonl.zst"

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

for i, record in enumerate(read_zst_file(file_path)):
    print(record)
    input()
