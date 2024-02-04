# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

from datasets import load_dataset
from distilabel.llm import InferenceEndpointsLLM
from distilabel.pipeline import pipeline
from distilabel.tasks import TextGenerationTask

if __name__ == "__main__":
    dataset = (
        load_dataset("teknium/OpenHermes-2.5", split="train[:100]")
        .rename_column("conversations", "input")
    )

    pipe = pipeline(
        "preference",
        "text-quality",
        generator=InferenceEndpointsLLM(
            endpoint_name_or_model_id="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",  # type: ignore
            task=TextGenerationTask(system_prompt="Act as a translator, converting english inputs into pure "
                                                  "mexican-dialect spanish."),
            prompt_format="chatml",
            max_new_tokens=4092,
            num_threads=16,
            temperature=1.2,
        ),
        max_new_tokens=256,
        num_threads=2,
        api_key="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",
        temperature=0.0,
    )

    start = time.time()
    dataset = pipe.generate(
        dataset,  # type: ignore
        num_generations=2,
        batch_size=1,
        checkpoint_strategy=True,
        display_progress_bar=True,
    )
    end = time.time()
    print("Elapsed", end - start)

    # Push to the HuggingFace Hub
    dataset.push_to_hub(
        "SiguienteGlobal/hermes-es",  # type: ignore
        split="train",
        private=True,
        token="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",
    )
