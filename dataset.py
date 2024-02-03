
import json

from datasets import load_dataset
from transformers import pipeline

def translator():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")


def map_object(obj, func):
    """ Function to map each value in the object through the lambda function """
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")

    mapped_obj = {}
    for key, value in obj.items():
        mapped_obj[key] = func(value)
    return mapped_obj


if __name__ == "__main__":
    pipe = translator()
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
    dataset = dataset.map(lambda x: dict(messages=map_object(dict(x["conversations"]), pipe)), remove_columns=["conversations"], batched=True)
    dataset.push_to_hub("SiguienteGlobal/OpenHermes-2.5-es")
