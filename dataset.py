import torch.distributed as dist
from transformers import MarianMTModel, pipeline, AutoTokenizer
from datasets import load_dataset
from sacremoses import MosesTokenizer, MosesPunctNormalizer
from accelerate import Accelerator

def setup_distributed():
    dist.init_process_group(backend='nccl')

def load_dataset_distributed(model, tokenizer):
    translator = pipeline("translation", model=model, max_length=10200, tokenizer=tokenizer)
    
    en = MosesTokenizer(lang='en')
    mpn = MosesPunctNormalizer()

    dataset = load_dataset("teknium/OpenHermes-2.5")
    dataset = dataset["train"]
    dataset = dataset.map(
        lambda col: dict(conversations=[process(chain, translator, en, mpn) for chain in col["conversations"]]), batched=True
    )
    return dataset

def process(message, translator, en: MosesTokenizer, mpn: MosesPunctNormalizer):
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
    accelerator = Accelerator(split_batches=True)
    accelerator.prepare_data_loader()
    setup_distributed()

    translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es", max_length=10200).to(accelerator.device)
    translator_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es", max_length=10200)

    dataset = load_dataset_distributed(translator_model, translator_tokenizer)
    dataset.push_to_hub("SiguienteGlobal/spanglang")

if __name__ == '__main__':
    main()
