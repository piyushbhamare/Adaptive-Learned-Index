#!/usr/bin/env python3
"""
validate_datasets.py — SOSD Dataset Validation Script

Validates all three SOSD uint64 binary datasets:
  books_200M_uint64, fb_200M_uint64, wiki_ts_200M_uint64

Checks:
  1. File existence
  2. File size (expected: 8 + count*8 bytes)
  3. Count header matches file size
  4. Keys are monotonically non-decreasing (sorted)
  5. SHA-256 checksum of first 1000 keys (for integrity)

Output: JSON validation report

Author: NLI Group 19, 2025-26
"""

import argparse
import hashlib
import json
import os
import struct
import sys


DATASETS = [
    "books_200M_uint64",
    "fb_200M_uint64",
    "wiki_ts_200M_uint64",
]

# Expected: 200M keys each
EXPECTED_COUNT = 200_000_000


def validate_file(path: str) -> dict:
    """Validate a single SOSD binary dataset."""
    result = {
        "path": path,
        "exists": False,
        "file_size_bytes": 0,
        "header_count": 0,
        "expected_size_bytes": 0,
        "size_ok": False,
        "monotonic": None,
        "min_key": None,
        "max_key": None,
        "first1000_sha256": None,
        "valid": False,
        "notes": [],
    }

    if not os.path.isfile(path):
        result["notes"].append("File not found")
        return result

    result["exists"] = True
    fsize = os.path.getsize(path)
    result["file_size_bytes"] = fsize

    with open(path, "rb") as f:
        # Read 8-byte count header
        raw = f.read(8)
        if len(raw) < 8:
            result["notes"].append("File too small for header")
            return result

        count = struct.unpack("<Q", raw)[0]
        result["header_count"] = count
        result["expected_size_bytes"] = 8 + count * 8

        # Check size (allow ±8 bytes for alignment padding)
        size_ok = abs(fsize - (8 + count * 8)) <= 8
        result["size_ok"] = size_ok
        if not size_ok:
            result["notes"].append(
                f"Size mismatch: file={fsize}, expected={8+count*8}"
            )

        # Checksum of first 1000 keys
        first_raw = f.read(1000 * 8)
        sha = hashlib.sha256(first_raw).hexdigest()
        result["first1000_sha256"] = sha

        # Sample keys for monotonicity check
        n_sample = min(count, 20_000)
        step = max(1, count // n_sample)
        sample_keys = []
        for i in range(0, count, step):
            f.seek(8 + i * 8)
            raw = f.read(8)
            if len(raw) == 8:
                sample_keys.append(struct.unpack("<Q", raw)[0])

        if sample_keys:
            result["min_key"] = sample_keys[0]
            result["max_key"] = sample_keys[-1]
            mono = all(
                sample_keys[i] <= sample_keys[i + 1]
                for i in range(len(sample_keys) - 1)
            )
            result["monotonic"] = mono
            if not mono:
                result["notes"].append("Keys are NOT monotonically sorted!")

    # Overall validity
    result["valid"] = (
        result["exists"]
        and result["size_ok"]
        and result["monotonic"] is True
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate SOSD datasets")
    parser.add_argument(
        "--data-dir", default="sosd_data", help="Directory containing dataset files"
    )
    parser.add_argument(
        "--out", default="results/dataset_validation.json",
        help="Output JSON file"
    )
    args = parser.parse_args()

    print("\n=== SOSD Dataset Validation ===\n")

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    all_results = []
    all_valid = True

    for name in DATASETS:
        path = os.path.join(args.data_dir, name)
        print(f"  Checking {name} ...", end="", flush=True)
        res = validate_file(path)
        all_results.append({"dataset": name, **res})

        status = "✓ VALID" if res["valid"] else "✗ INVALID"
        print(f" {status}")
        print(f"    size={res['file_size_bytes']:,} bytes  count={res['header_count']:,}")
        if res["min_key"] is not None:
            print(f"    keys: [{res['min_key']:,} .. {res['max_key']:,}]  monotonic={res['monotonic']}")
        if res["first1000_sha256"]:
            print(f"    sha256(first1000)={res['first1000_sha256'][:32]}...")
        if res["notes"]:
            for note in res["notes"]:
                print(f"    [!] {note}")
        print()

        if not res["valid"]:
            all_valid = False

    report = {
        "datasets": all_results,
        "all_valid": all_valid,
        "data_dir": args.data_dir,
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Validation report written to: {args.out}")

    if all_valid:
        print("\n✓ All datasets are valid. Ready to benchmark.\n")
        return 0
    else:
        print("\n✗ One or more datasets failed validation.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
