import json

from datasets import load_dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

def map_object(obj, func):
    """ Function to map each value in the object through the lambda function """
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")
    def message(row, pipe):
        return {"role":"user" if row["from"] == "human" else "assistant", "content":pipe(row["value"])[0]["translation_text"]}
    convo = [
        message(row, func) for row in obj
    ]
    return convo


def map_row(pipeline):
    def mapper(row):
        return dict(
            conversations=[
                map_object(x, pipeline) for x in row["conversations"]
            ]
        )

    return mapper


if __name__ == "__main__":
    accelerator = Accelerator()
    pipe = pipeline(
        "text-generation",
        model=AutoModelForCausalLM.from_pretrained("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", device_map="auto"),
        tokenizer=AutoTokenizer.from_pretrained("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"),
        model_kwargs={"prompt":"Translate to spanish"}
    )
    transform = map_row(
        pipe("translation", model="Helsinki-NLP/opus-mt-en-es", max_length=32000)
    )
    def batch_transform(vals):
        try:
            return transform(vals)
        except:
            pass
    dataset = load_dataset("teknium/OpenHermes-2.5")
    dataset = dataset.map(batch_transform, batched=True)
    dataset.push_to_hub("SiguienteGlobal/hermes-es")
