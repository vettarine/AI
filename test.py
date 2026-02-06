import kagglehub

# Download latest version
path = kagglehub.dataset_download("sumanthvrao/fakenewsdataset")

print("Path to dataset files:", path)