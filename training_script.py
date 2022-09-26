from spacy.cli.train import train as spacy_train
config_path = "/Users/kspicer/Desktop/spacy_textcat/config.cfg"
output_model_path = "/Users/kspicer/Desktop/spacy_textcat/output"
spacy_train(
    config_path,
    output_path=output_model_path,
    overrides={
        "paths.train": "train.spacy",
        "paths.dev": "valid.spacy",
    },
)
