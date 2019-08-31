import json
import typing
import os

def read_json_from_file(filepath: str):
    with open(filepath, 'r') as fp:
        return json.load(fp)

def _ensure_dir_exists(filepath:str):
    dst_dir = os.path.dirname(filepath)
    os.makedirs(dst_dir, exist_ok=True)

def write_json_to_file(obj, filepath:str):
    _ensure_dir_exists(filepath)
    with open(filepath, 'w') as fp:
        json.dump(obj, fp)