import torch
import json


class MetricsLogger:
    """
    A class that logs metrics to a json file.
    The keys in the json file represents a section, and the values are a list of metrics for that section.

    It can be something like:
        {
            "train": [
                {
                    "train_step": 1,
                    "metric_1": "value_1"
                },
                {
                    "train_step": 2,
                    "metric_1": "value2"
                },
            ],
            "val": [
                {
                    "train_step": 1,
                    "metric_2": "value_3",
                    "metric_3": "value_4",
                },
                {
                    "train_step": 2,
                    "metric_2": "value_5",
                    "metric_3": "value_6"
                }
            ],
            "section_3": [
                "value_7",
                "value_8",
                "value_9"
            ]
        }
    """  # noqa: E501

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._data = {}
        self._save_data()

    def _load_data(self):
        with open(self.file_path, "r") as file:
            data = json.load(file)
        return data

    def _is_json_serializable(self, value):
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False

    def _convert_to_serializable(self, value):
        if isinstance(value, (int, float, str)):
            return value
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                return value.item()
            elif value.dim() == 1:
                return value.tolist()
            else:
                return value.numpy().tolist()
        elif isinstance(value, list):
            return [self._convert_to_serializable(item) for item in value]
        elif isinstance(value, dict):
            return {
                key: self._convert_to_serializable(val) for key, val in value.items()
            }
        else:
            return str(value)

    def _save_data(self):
        with open(self.file_path, "w") as file:
            json.dump(self._data, file, indent=4)

    def log(self, section, value):
        if section not in self._data:
            self._data[section] = []

        if not self._is_json_serializable(value):
            value = self._convert_to_serializable(value)

        self._data[section].append(value)
        self._save_data()
