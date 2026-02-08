"""Очистка неиспользуемых моделей из кеша HuggingFace."""
from huggingface_hub import scan_cache_dir

cache = scan_cache_dir()
to_delete = [
    "gpt2",
    "distilgpt2",
    "gpt2-medium",
    "timm/inception_resnet_v2.tf_in1k",
    "sberbank-ai/rugpt3small_based_on_gpt2",
]

hashes = []
for repo in cache.repos:
    if repo.repo_id in to_delete:
        for rev in repo.revisions:
            hashes.append(rev.commit_hash)
            print(f"  Delete: {repo.repo_id} ({repo.size_on_disk / 1e6:.0f} MB)")

if not hashes:
    print("Nothing to delete")
else:
    strategy = cache.delete_revisions(*hashes)
    print(f"Will free: {strategy.expected_freed_size / 1e9:.2f} GB")
    strategy.execute()
    print("Done")

# Check remaining
cache2 = scan_cache_dir()
print(f"\nRemaining cache: {cache2.size_on_disk / 1e9:.2f} GB")
for r in cache2.repos:
    print(f"  {r.repo_id}: {r.size_on_disk / 1e9:.2f} GB")
