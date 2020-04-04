import json
import os


class Task:
    @staticmethod
    def check_format(filepath: str) -> bool:
        raise NotImplementedError


class NER(Task):
    @staticmethod
    def check_format(filepath: str) -> bool:
        try:
            with open(filepath) as f:
                for line in f:
                    json.loads(line.rstrip(os.linesep))
        except json.JSONDecodeError as e:
            raise e
