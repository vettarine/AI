import kagglehub

path = kagglehub.dataset_download(
    "jessicali9530/stanford-dogs-dataset"
)

print("Path to dataset:", path)