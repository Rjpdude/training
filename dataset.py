from datasets import load_dataset
from distilabel.llm import OpenAILLM
from distilabel.pipeline import pipeline
from distilabel.tasks import TextGenerationTask

dataset = (
    load_dataset("argilla/dpo-mix-7k", split="train[:05]").rename_column("prompt", "input")
)

# Create a `Task` for generating text given an instruction.
task = TextGenerationTask()

# Create a `LLM` for generating text using the `Task` created in
# the first step. As the `LLM` will generate text, it will be a `generator`.
generator = OpenAILLM(task=task, max_new_tokens=512, api_key="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")

# Create a pre-defined `Pipeline` using the `pipeline` function and the
# `generator` created in step 2. The `pipeline` function will create a
# `labeller` LLM using `OpenAILLM` with the `UltraFeedback` task for
# instruction following assessment.
pipeline = pipeline("preference", "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", generator=generator)

dataset = pipeline.generate(dataset)
