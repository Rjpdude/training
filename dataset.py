import torch.distributed as dist
from transformers import pipeline
from datasets import load_dataset
from sacremoses import MosesTokenizer, MosesPunctNormalizer
from accelerate import Accelerator

def load_dataset_distributed(model_id):
    translator = pipeline("translation", model_id)

    en = MosesTokenizer(lang='en')
    mpn = MosesPunctNormalizer()

    dataset = load_dataset("teknium/OpenHermes-2.5")
    dataset = dataset["train"]
    dataset = dataset.map(
        lambda col: dict(conversations=[process(chain, translator, en, mpn) for chain in col["conversations"]]),
        batched=True
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
    @accelerator.on_main_process
    def init():
        dist.init_process_group(backend='nccl')

    init()
    accelerator.wait_for_everyone()
    dataset = load_dataset_distributed("Helsinki-NLP/opus-mt-en-es")
    accelerator.wait_for_everyone()
    dataset.push_to_hub("SiguienteGlobal/spanglang")


if __name__ == '__main__':
    main()
