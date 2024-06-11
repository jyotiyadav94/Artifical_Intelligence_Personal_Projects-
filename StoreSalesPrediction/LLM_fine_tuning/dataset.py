import json
import os
from pathlib import Path
import datasets

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """
Custom Question Answer Dataset: This dataset consists of question-answer pairs formatted as JSON. Each example includes an input text representing a non-factoid question and an output text representing the corresponding answer.

The dataset is structured with the following features:
- id: Unique identifier for each example.
- input_text: The non-factoid question.
- output_text: The corresponding answer.

This dataset is intended for training and evaluating models on non-factoid question answering tasks.
"""

class CustomDataset(datasets.GeneratorBasedBuilder):
    """Custom Non-Factoid Question Dataset"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {   "id": datasets.Value("string"),
                    "input_text": datasets.Value("string"),
                    "output_text": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        # Specify the path to the dataset file
        filepath = "/content/drive/Shareddrives/dataset/clients/research/formatted_data.json"
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": filepath}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                for idx, example in enumerate(data):
                    yield idx, {
                        "input_text": example["input_text"],
                        "output_text": example["output_text"],
                    }
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file '{filepath}' not found.")
