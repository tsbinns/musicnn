import os
import pickle
import shutil

from musicnn.extractor import extractor
from musicnn import configuration as config


DATA_PATH = os.path.join("tests", "data")
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


def main():
    """Get outputs from TF v1 API for regression tests with TF v2 API."""
    music_file = os.path.join("audio", "TRWJAZW128F42760DD_cropped.mp3")
    models = config.MODELS
    for model in models:
        taggram, labels, features = extractor(
            file_name=music_file,
            model=model,
            input_length=3.0,
            input_overlap=1.0,
            extract_features=True,
        )
        outputs = dict(taggram=taggram, labels=labels, features=features)
        with open(os.path.join(DATA_PATH, f"{model}_outputs.pkl"), "wb") as file:
            pickle.dump(outputs, file)
    shutil.make_archive(DATA_PATH, "zip", DATA_PATH)
    shutil.rmtree(DATA_PATH)


if __name__ == "__main__":
    main()
