import torch
import torch.distributed as dist
from transformers import MarianMTModel, pipeline, AutoTokenizer
from transformers.tools.evaluate_agent import translator
from datasets import load_dataset
from sacremoses import MosesTokenizer, MosesPunctNormalizer

def setup_distributed():
    dist.init_process_group(backend='nccl')

def load_dataset_distributed():
    en = MosesTokenizer(lang='en')
    mpn = MosesPunctNormalizer()
    dataset = load_dataset("teknium/OpenHermes-2.5")
    dataset = dataset["train"]
    dataset = dataset.map(
        lambda col: dict(conversations=[process(chain, en, mpn) for chain in col["conversations"]]), batched=True
    )
    return dataset

def process(message, en: MosesTokenizer, mpn: MosesPunctNormalizer):
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

def main():
    setup_distributed()
    torch.cuda.set_device(torch.distributed.get_rank())

    translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es", max_length=10200).to("cuda")
    translator_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es", max_length=10200)
    translator = pipeline("translation", model=translator_model, max_length=10200, tokenizer=translator_tokenizer)

    dataset = load_dataset_distributed()
    dataset.push_to_hub("SiguienteGlobal/spanglang")

if __name__ == '__main__':
    main()
