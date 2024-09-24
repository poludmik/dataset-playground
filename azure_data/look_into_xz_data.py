import json
import lzma

file_path = "czech_llm_data/czech-llm-dataset-complete/syn/v9/raw/syn_v9.xz"

def read_xz_file_in_chunks(file_path, chunk_size=8192):
    with lzma.open(file_path, 'rt') as f:
        buffer = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break  # End of file
            buffer += chunk
            # Assuming the content has JSON objects or arrays separated by some delimiter
            # Let's say it's JSON objects, separated by `\n` or `}`
            while '}' in buffer:  # Assuming JSON objects end with '}'
                try:
                    record, buffer = buffer.split('}', 1)
                    yield json.loads(record + '}')  # Complete the JSON object
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

# Process the file in chunks
for i, record in enumerate(read_xz_file_in_chunks(file_path)):
    print(record)
    input()  # Pause to inspect each record