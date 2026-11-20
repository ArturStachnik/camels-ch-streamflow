"""Download CAMELS-CH from Zenodo and verify the MD5 checksums.

Source: Hoege et al. (2023), ESSD 15, 5755-5784. Concept DOI 10.5281/zenodo.7784632.
This script pins version 0.9 (record 15025258, published 2025-03-14, CC-BY-4.0).
The dataset is not redistributed with this repository.

Usage: python scripts/download_data.py [--dest data/raw]
"""
import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

RECORD = "15025258"
FILES = {
    "camels_ch.zip": "04f909d9904375647d030c4ab8ddfdbe",
    "camels_ch_data_description.pdf": "afcf00ebdd82359eea5062ae4a230953",
}
EXTRACT_DIR = "camels_ch_v0.9"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    for name, expected in FILES.items():
        target = args.dest / name
        if not target.exists():
            url = f"https://zenodo.org/api/records/{RECORD}/files/{name}/content"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, target)
        got = md5(target)
        if got != expected:
            raise SystemExit(f"MD5 mismatch for {name}: expected {expected}, got {got}")
        print(f"{name}: md5 ok ({got})")

    out = args.dest / EXTRACT_DIR
    if not out.exists():
        out.mkdir()
        with zipfile.ZipFile(args.dest / "camels_ch.zip") as z:
            z.extractall(out)
    print(f"extracted to {out}")


if __name__ == "__main__":
    main()
