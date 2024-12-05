import json
import os


def make_serializable(data):
    """
    Recursively converts non-serializable fields in a dictionary into strings.

    Args:
        data (dict): The input dictionary to process.

    Returns:
        dict: A dictionary with all non-serializable fields converted to strings.
    """
    if isinstance(data, dict):
        return {key: make_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [make_serializable(item) for item in data]
    try:
        json.dumps(data)  # Test if the value is serializable
        return data
    except (TypeError, ValueError):
        return str(data)


def save_dict_to_json(data: dict, file_name: str, dir_path: str = "results/") -> None:
    """
    Saves a dictionary as a JSON file to the specified directory,
    handling non-serializable fields by converting them to strings.

    Args:
        data (dict): The dictionary to save.
        file_name (str): The name of the JSON file (should include .json extension).
        dir_path (str): The directory path where the file will be saved. Defaults to 'results/'.

    Raises:
        ValueError: If file_name does not end with .json.
        IOError: If there is an issue creating the file or writing to disk.
    """
    if not file_name.endswith(".json"):
        raise ValueError("file_name must have a .json extension.")

    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)

    # Convert non-serializable fields
    serializable_data = make_serializable(data)

    # Construct the full file path
    file_path = os.path.join(dir_path, file_name)

    # Write the dictionary to a JSON file
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(serializable_data, file, indent=4, ensure_ascii=False)
        print(f"Dictionary successfully saved to {file_path}")
    except IOError as e:
        print(f"An error occurred while writing to {file_path}: {e}")
