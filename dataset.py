import json

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, load_tool, AutoModelForCausalLM, LocalAgent

model = AutoModelForCausalLM.from_pretrained("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", device_map="auto",
                                             torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",)


tokenizer = AutoTokenizer.from_pretrained("meetkai/functionary-medium-v2.2")
agent = LocalAgent(model, tokenizer)
capy = load_dataset("argilla/distilabel-capybara-dpo-9k-binarized", split="train")


def convert(row):
    copy = dict.copy(row)
    copy.update(
        chosen=agent.run(
            f"Convert the following JSON conversation into spanish while maintaining the JSON structure: {row['chosen']}")
    )
    return copy

capy.map(convert, batched=True).to_json("converted.json")

# capy = capy.filter(
#     lambda r: r["rating_chosen"] >= 4
# )
#
# capy = capy.map(lambda r: {"messages": len(r["chosen"])}).filter(lambda r: r["messages"] < 18)


# def chatml_format(example):
#     example["chosen"] = agent.chat(
#         f"Translate the following from spanish into a mexican dialect of spanish: {example['chosen']}")
#     return example
#     # system = ""
#     # # get everything except the last message as input
#     # prompt = tokenizer.apply_chat_template(example["chosen"][:-1], tokenize=False, add_generation_prompt=True)
#     # # get the last assistant responses
#     # chosen = example["chosen"][-1]["content"] + "</s>"
#     # rejected = example["rejected"][-1]["content"] + "</s>"
#     #
#     # return {
#     #     "prompt": agent.run_prompt_template(),
#     #     "chosen": agent.run(chosen),
#     #     "rejected": agent.run(rejected),
#     # }
#
#
# # Save columns
# original_columns = capy.column_names
#
# # Format dataset
# capy = capy.map(chatml_format, remove_columns=original_columns, batched=True)

# capy.to_json("formatted.json")
