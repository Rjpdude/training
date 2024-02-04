from dataclasses import dataclass
from transformers import AutoModelForCausalLM, MarianMTModel, pipeline, AutoTokenizer, BitsAndBytesConfig
from transformers.tools.evaluate_agent import translator
from datasets import load_dataset
from sacremoses import MosesTokenizer, MosesPunctNormalizer

en = MosesTokenizer(lang='en')
mpn = MosesPunctNormalizer()


@dataclass
class Source:
    path: str
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def init(self):
        from transformers import set_seed

        set_seed(91)

        config = BitsAndBytesConfig(
            load_in_8bit=False,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.path, device_map="auto", quantization_config=config,
                                                          attn_implementation="flash_attention_2", max_length=4092)
        return self

    def generate(self, input):
        model_inputs = self.tokenizer(input, padding=True, return_tensors="pt").input_ids.cuda()
        generated_ids = self.model.generate(**model_inputs)
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output


def process(message, model):
    try:
        queue = []
        for chain in message:
            role = "user" if chain["from"] == "human" else "assistant"
            pending = en.tokenize(chain["value"], return_str=True)
            output = model.generate(f"Translate the following into Spanish: {pending}")
            res = dict(role=role, content=mpn.normalize(output))
            print(res)
            queue.append(res)
        return queue
    except:
        pass


if __name__ == "__main__":
    model = Source(path="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT").init()
    dataset = load_dataset("teknium/OpenHermes-2.5")
    dataset = dataset["train"]
    dataset = dataset.map(
        lambda col: dict(conversations=[process(chain, model) for chain in col["conversations"]]), batched=True
    )
    dataset.push_to_hub("SiguienteGlobal/spanglang")
