from transformers import MarianMTModel, pipeline, AutoTokenizer
from transformers.tools.evaluate_agent import translator
from datasets import load_dataset
from sacremoses import MosesTokenizer, MosesPunctNormalizer
import torch

translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es", max_length=10200).to("cuda")
translator_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es", padding='max_length', truncation=True)
translator = pipeline("translation", model=translator_model, max_length=10200, tokenizer=translator_tokenizer)

en = MosesTokenizer(lang='en')
mpn = MosesPunctNormalizer()

def process(message):
    try:
        queue = []
        for chain in message:
            role = "user" if chain["from"] == "human" else "assistant"
            code = translator(en.tokenize(chain["value"], return_str=True))
            res = dict(role=role, content=mpn.normalize(code[0]["translation_text"]))
            queue.append(res)
        return queue
    except:
        pass

dataset = load_dataset("teknium/OpenHermes-2.5")
dataset = dataset["train"]
dataset = dataset.map(
    lambda col: dict(conversations=[process(chain) for chain in col["conversations"]]), batched=True
)
dataset.push_to_hub("SiguienteGlobal/spanglang")
