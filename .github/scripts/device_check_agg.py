#!/usr/bin/env python3

import json
import os
from glob import glob


def main():
    """
    1. Recursively find partial_results.json in the downloaded artifacts folder.
    2. Merge them into a single JSON array (merged_results.json).
    3. Convert that JSON array into a final markdown file (final_results.md).
    4. Print the final markdown table to stdout for logs.
    """

    # 1. Find partial JSON files (downloaded via actions/download-artifact)
    partial_files = glob(
        os.path.join("all_artifacts", "**", "partial_results.json"), recursive=True
    )
    results_list = []

    # 2. Read and collect each partial JSON
    for file_path in partial_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            results_list.append(data)

    # 3. Write a merged JSON for debugging/visibility
    with open("merged_results.json", "w") as f:
        json.dump(results_list, f, indent=2)

    # 4. Convert the merged JSON list to a Markdown table
    header = "# Final Merged Conversion Testing Results\n\n"
    header += "| Conversion Mode | Device | Status |\n"
    header += "|-----------------|--------|--------|\n"

    rows = ""
    for item in results_list:
        mode = item.get("mode", "N/A")
        device = item.get("device", "N/A")
        status = item.get("status", "N/A")
        rows += f"| {mode} | {device} | {status} |\n"

    with open("final_results.md", "w") as out:
        out.write(header + rows)

    # 5. Print final table for logs
    print(header + rows)


if __name__ == "__main__":
    main()
