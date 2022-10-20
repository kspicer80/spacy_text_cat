import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from spacy.cli.train import train as spacy_train

config_path = "config.cfg"
output_model_path = "./output"
spacy_train(
    config_path,
    output_path=output_model_path,
    overrides={
        "paths.train": "train_full.spacy",
        "paths.dev": "valid_full.spacy",
    },
)
