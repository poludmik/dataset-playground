from azure.storage.blob import BlobServiceClient, ContainerClient
import os

# Replace with your connection string
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')  # Load from environment variable
container_name = "promethist-data"
folder_name = "czech-llm-dataset-complete/commoncrawl/"  # Starting with the "commoncrawl" folder
local_path = "czech_llm_data"  # Local path to save the files

# Pattern to match: "commoncrawl/*/cleaned-deduplicated-url_deduplicated/"
target_subfolder_suffix = "cleaned-deduplicated-url_deduplicated/"

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Create the ContainerClient object
container_client = blob_service_client.get_container_client(container_name)

# List all blobs in the commoncrawl folder
blob_list = container_client.list_blobs(name_starts_with=folder_name)

print(f"Downloading files from {folder_name} that match the pattern '*/cleaned-deduplicated-url_deduplicated/'")

for blob in blob_list:
    # Check if the blob's path matches the "commoncrawl/*/cleaned-deduplicated-url_deduplicated/" pattern
    if target_subfolder_suffix in blob.name:
        # Make sure there's exactly one folder between "commoncrawl/" and the suffix
        # For example: commoncrawl/some-folder/cleaned-deduplicated-url_deduplicated/
        sub_path = blob.name[len(folder_name):]
        parts = sub_path.split("/")

        if len(parts) >= 2 and parts[-2] == target_subfolder_suffix.strip('/'):
            # Create a local file path, matching the blob structure
            local_file_path = os.path.join(local_path, blob.name)
            
            # Check if the file already exists
            if os.path.exists(local_file_path):
                print(f"File {local_file_path} already exists. Skipping download.")
                continue
            
            # If the file does not exist, download it
            if not os.path.exists(os.path.dirname(local_file_path)):
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the blob
            print(f"Downloading {blob.name}...")
            with open(local_file_path, "wb") as download_file:
                download_file.write(container_client.download_blob(blob.name).readall())
            print(f"Downloaded {blob.name}")
    else:
        print(f"Skipping {blob.name} (does not match target folder structure)")

print("Download complete.")
