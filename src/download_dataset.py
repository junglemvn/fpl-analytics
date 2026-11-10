"""Download a versioned dataset copy without replacing existing raw inputs"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPOSITORY = "Toibinguz/epl-premier-league-dataset"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data/epl-premier-league-dataset-updated"


def download_dataset(repo_id=REPOSITORY, output_dir=OUTPUT_DIR):
    """Pin the latest Hub revision, download its files and verify reported sizes"""
    info = HfApi().dataset_info(repo_id, files_metadata=True, timeout=30)
    destination = Path(output_dir) / info.sha
    snapshot_download(
        repo_id, repo_type="dataset", revision=info.sha, local_dir=destination,
        cache_dir=Path(output_dir) / ".cache", local_dir_use_symlinks=False,
        resume_download=True, max_workers=4,
    )
    invalid = [
        file.rfilename for file in info.siblings
        if not (destination / file.rfilename).is_file()
        or file.size is not None and (destination / file.rfilename).stat().st_size != file.size
    ]
    if invalid:
        raise RuntimeError(f"Missing or incomplete files: {invalid}")
    manifest = {
        "repository": repo_id, "revision": info.sha,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [{"path": file.rfilename, "bytes": (destination / file.rfilename).stat().st_size}
                  for file in info.siblings],
    }
    (destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded and checked {len(info.siblings)} files: {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPOSITORY)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        download_dataset(args.repo, args.output)
    except Exception as error:
        parser.exit(1, f"Download failed: {error}\n")
