import pathlib as pl
from typing import Dict, Union, Literal, Any, Optional
import json
from tqdm import tqdm
import pickle
from PIL import Image



def load_json(
        data_dir: str,
        json_name: str = "dataset.json") -> Optional[Dict[str, Any]]:
    r"""Function to load the respective dataset.json of the specified file"""

    dataset_path = pl.Path(data_dir)
    if dataset_path.is_dir():
        path_to_json = pl.Path(dataset_path, json_name)
        try:
            with open(path_to_json, 'r') as _json:
                json_file = json.load(_json)
            return json_file
        except FileNotFoundError:
            return None

    else:
        raise AttributeError(f"Specified path '{str(dataset_path)} is not a directory.'")