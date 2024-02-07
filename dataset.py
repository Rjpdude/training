# This script is used to tokenize the dataset ahead for packing
# It is not necessary to run this script if you are using the PackedDataset class.
# Beware: This uses Functionary prompting template to tokenize.
import json
import os
from json import JSONEncoder, JSONDecoder

import typer
from datasets import load_dataset
from transformers import LlamaTokenizer, MixtralForCausalLM, BitsAndBytesConfig, set_seed

from functionary.prompt_template import get_prompt_template_by_version
from functionary.train.custom_datasets import PackedDataset


def main(
        pretrained_path: str = typer.Option(default="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT"),
        data_path: str = typer.Option(default="dataset.json"),
        save_folder: str = typer.Option(default="training"),
        data_type: str = typer.Option(default="train"),  # train/validation
        template_version: str = typer.Option(default="v2"),
        max_length: int = typer.Option(4096),
):
    """Tokenize the dataset ahead for packing

    Args:
        pretrained_path (str): pretrained model to use
        data_path (str): path to .jsonl file
        save_folder (str): where to save (the output_dir in training)
        data_type (str): one of: "train" or "validation"
        template_version: v1 or v2
        max_length (int, optional): max_length for tokenizer
    """
    assert data_type in ["train", "validation"]
    set_seed(8)
    prompt_template = get_prompt_template_by_version(template_version)

    config = BitsAndBytesConfig(load_in_4bit=True, load_in_8bit=False, bnb_4bit_use_double_quant=True)
    model = MixtralForCausalLM.from_pretrained(pretrained_path, device_map="auto", quantization_config=prompt_template)
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train", keep_in_memory=True, trust_remote_code=True)
    tokenizer = LlamaTokenizer.from_pretrained(
        pretrained_path,
        model_max_length=max_length,
        legacy=True,
    )

    tools = [
        {
            "type": "function",
            "function": {
                {
                    "type": "function",
                    "function": {
                        "name": "buscar_usuario",
                        "description": "Buscar los datos actuales del usuario.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "user_id": {
                                    "type": "string",
                                    "description": "El identificador unico de la cuenta del usuario."
                                },
                                "clave_fastpass": {
                                    "type": "number",
                                    "description": "La clave establicido entre el usuario y su asistente IA para permitir acceso."
                                }
                            },
                            "required": [
                                "user_id", "clave_fastpass"
                            ]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "crear_imagen",
                        "description": "A travez de un prompt, genera una imagen vivida (aviso: puede demorar hasta 60 segundos)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "Un prompt bastante detallada de como se vera la imagen."
                                }
                            },
                            "required": [
                                "prompt"
                            ]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "voz",
                        "description": "Responder con voz. Con este funccion puedas hablar al usuario.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "El texto que quieras mandar al usuario - ex. 'Estoy muy bien, y tu?'"
                                }
                            },
                            "required": [
                                "prompt"
                            ]
                        }
                    }
                },
            }
        }
    ]

    def generator(user_input):
        model_inputs = tokenizer([user_input], return_tensors="pt", padding_side="left", padding=True,
                                 model_max_length=max_length).to("cuda")
        generated_ids = model.generate(**model_inputs, do_sample=True)
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def generate_tools(convo):
        template = tokenizer.apply_chat_template([
            {"role": "system",
             "content": f"Assess the following conversation and generate a set of openai-formatted tools given the "
                        f"conversations context. Here is an example format: {json.dumps(tools)}"},
            {"role": "user", "content": convo},
        ])
        return generator(template)

    def translate(text):
        template = tokenizer.apply_chat_template([
            {"role": "system",
             "content": f"Given a user's english input, output a translated spanish version with careful precision and "
                        f"attention to detail and subtleties."},
            {"role": "user", "content": text},
        ])
        return generator(template)

    def preprocess(convoset):
        def convert(convo):
            convo = {
                "messages": [
                    {"role": "user", "content": translate(convo[0]["value"])},
                    {"role": "assistant", "content": translate(convo[1]["value"])}
                ]
            }
            toolset = generate_tools(json.dumps(convo))
            convo["tools"] = toolset
            return toolset

        return [dict(messages=convert(conversations)) for conversations in convoset["conversations"]]

    generated = dataset["train"].map(preprocess)
    generated.to_json(data_path)

    tokenizer.pad_token = tokenizer.eos_token
    added_tokens = prompt_template.get_additional_tokens()
    special_tokens = {"additional_special_tokens": added_tokens}
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    print("number of added tokens: ", num_new_tokens)

    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]

    keep_assistant_prefix = True if data_type == "train" else False
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    cached_folder = f"{save_folder}/{data_type}_cached"
    if not os.path.exists(cached_folder):
        os.mkdir(cached_folder)

    print("number of items: ", len(raw_data))
    ds = PackedDataset(
        raw_data,
        tokenizer,
        cached_folder=cached_folder,
        ignore_cached=False,
        keep_assistant_prefix=keep_assistant_prefix,
        use_flash_attention=True,
        pack_length=max_length,
    )
    ds.stat()


if __name__ == "__main__":
    typer.run(main)
