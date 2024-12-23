#!/usr/bin/env python3

import json
import os
from glob import glob


def pad_string(string, length):
    """Pads a string with spaces to ensure it matches the desired length."""
    return string + " " * (length - len(string))


def main():
    """
    1. Recursively find all *.json files in the downloaded artifacts folder.
    2. Merge them into a single JSON array (merged_results.json).
    3. Dynamically pad rows based on the longest conversion mode.
    4. Convert the results into a markdown table.
    """

    # 1. Find all JSON files (downloaded via actions/download-artifact)
    partial_files = glob(os.path.join("all_artifacts", "**", "*.json"), recursive=True)
    results_list = []

    # 2. Read and collect data from each JSON file
    for file_path in partial_files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                mode = data.get("mode", "N/A")
                device = data.get("device", "N/A")
                status = data.get("status", "❌")
                results_list.append({"mode": mode, "device": device, "status": status})
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON file {file_path}")

    # Ensure results_list is not empty
    if not results_list:
        print("Error: No valid JSON files found in 'all_artifacts'.")
        return

    # 3. Organize results into rows by mode
    aggregated_results = {}
    for result in results_list:
        mode = result["mode"]
        device = result["device"]
        status = result["status"]

        if mode not in aggregated_results:
            aggregated_results[mode] = {"CUDA": "❌", "CPU": "❌", "MPS": "?"}
        aggregated_results[mode][device.upper()] = status

    # Determine the longest mode for padding
    max_mode_length = max(len(mode) for mode in aggregated_results.keys())
    mode_column_width = (
        max(max_mode_length, len("Conversion Option")) + 2
    )  # Add padding

    # 4. Write a merged JSON for debugging/visibility
    with open("merged_results.json", "w") as f:
        json.dump(aggregated_results, f, indent=2)

    # 5. Convert the aggregated results to a Markdown table
    header = f"| ID  | {'Conversion Option'.ljust(mode_column_width)} | CUDA  | CPU   | MPS  |\n"
    header += f"|-----|{'-' * mode_column_width}|-------|-------|------|\n"

    rows = ""
    for idx, (mode, statuses) in enumerate(aggregated_results.items(), start=1):
        id_col = pad_string(str(idx), 4)
        mode_col = pad_string(mode, mode_column_width)
        cuda_col = pad_string(statuses.get("CUDA", "❌"), 7)
        cpu_col = pad_string(statuses.get("CPU", "❌"), 7)
        mps_col = pad_string(statuses.get("MPS", "?"), 6)

        rows += f"| {id_col} | {mode_col} | {cuda_col} | {cpu_col} | {mps_col} |\n"

    final_table = header + rows

    # Write the final markdown file
    with open("final_results.md", "w") as out:
        out.write(final_table)

    # 6. Print final table for logs
    print(final_table)


if __name__ == "__main__":
    main()
