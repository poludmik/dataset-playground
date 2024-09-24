from azure.storage.blob import BlobServiceClient, ContainerClient
import os

# Replace with your connection string
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')  # Load from environment variable
container_name = "promethist-data"
folder_name = "czech-llm-dataset-complete"  # The folder (prefix)
local_path = "czech_llm_data"  # Local path to save the files

# Folders to exclude
exclude_folders = [
    "commoncrawl",
    "culturax/raw",
    "hplt/v1.2/raw",
    "hplt/v1.2/cleaned",
    "hplt/v1.1/raw-deduplicated"
]

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Create the ContainerClient object
container_client = blob_service_client.get_container_client(container_name)

# List all blobs in the folder (prefix)
blob_list = container_client.list_blobs(name_starts_with=folder_name)

print(f"Downloading files from {folder_name}, excluding {', '.join(exclude_folders)}")

for blob in blob_list:
    # Check if blob is part of any of the excluded folders
    if not any(exclude in blob.name for exclude in exclude_folders):
        # Create a local file path, matching the blob structure
        local_file_path = os.path.join(local_path, blob.name)
        if not os.path.exists(os.path.dirname(local_file_path)):
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the blob
            print(f"Downloading {blob.name}...")
            with open(local_file_path, "wb") as download_file:
                download_file.write(container_client.download_blob(blob.name).readall())
            print(f"Downloaded {blob.name}")
    else:
        print(f"Skipping {blob.name} (excluded folder)")

print("Download complete.")
