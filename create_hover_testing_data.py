"""
Script to create hover-testing-data.json from the HoVer training dataset.

Transformations:
  - Remove: supporting_facts, hpqa_id
  - Rename: label -> ground_truth, num_hops -> hop_level
"""

import json
import os

INPUT_FILE = os.path.join("data", "benchmarks", "hover_train_release_v1.1 (2).json")
OUTPUT_FILE = os.path.join("data", "benchmarks", "hover-testing-data.json")


def transform_entry(entry: dict) -> dict:
    """Transform a single HoVer entry."""
    return {
        "id": entry["uid"],
        "claim": entry["claim"],
        "ground_truth": entry["label"],
        "hop_level": entry["num_hops"],
    }


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {INPUT_FILE}")

    transformed = [transform_entry(e) for e in data[:100]]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(transformed, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(transformed)} entries (top 100) to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
