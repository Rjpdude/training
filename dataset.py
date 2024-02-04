import os

from datasets import load_dataset
from distilabel.llm import LLMPool, ProcessLLM, vLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask, UltraFeedbackTask
from vllm import LLM


def load_notus(task):
    llm = LLM(model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")
    return vLLM(model=llm, task=task, max_new_tokens=512, prompt_format="notus", tensor_parallel_size=4, trust_remote_code=True)


def load_zephyr(task):
    llm = LLM(model="cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser")
    return vLLM(model=llm, task=task, max_new_tokens=512, prompt_format="zephyr", tensor_parallel_size=4, trust_remote_code=True)


def load_openai(task):
    from distilabel.llm import OpenAILLM

    return OpenAILLM(
        model="gpt-3.5-turbo",
        task=task,
        api_key=os.getenv("OPENAI_API_KEY", "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"),
        max_new_tokens=512,
    )


if __name__ == "__main__":
    dataset = (
        load_dataset("teknium/OpenHermes-2.5", split="train")
        .rename_column("conversations", "input")
    )

    pipeline = Pipeline(
        generator=ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_zephyr),
        labeller=ProcessLLM(
            task=UltraFeedbackTask.for_instruction_following(), load_llm_fn=load_openai
        ),
    )

    dataset = pipeline.generate(
        dataset=dataset,  # type: ignore
        num_generations=3,
        batch_size=5,
    )
