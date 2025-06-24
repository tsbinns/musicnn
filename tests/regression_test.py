import os
import pickle
import zipfile

import numpy as np

from musicnn.extractor import extractor
from musicnn import configuration as config

DATA_PATH = os.path.join("tests", "data")
if not os.path.exists(DATA_PATH):
    with zipfile.ZipFile(os.path.join(DATA_PATH + ".zip"), "r") as zipped:
        zipped.extractall(DATA_PATH)


def test_regression_extractor():
    """Regression test for comparing results from old TF v1 API with new TF v2 API.

    Regression data was created using the script `tests/create_regression_data.py` from
    the `tf_v1_api` branch.
    """
    music_file = os.path.join("audio", "TRWJAZW128F42760DD_cropped.mp3")
    models = config.MODELS
    for model in models:
        print(f"Regression test for model '{model}'.")

        # Get results using this toolbox version (TF v2 API)
        taggram, labels, features = extractor(
            file_name=music_file,
            model=model,
            input_length=3.0,
            input_overlap=1.0,
            extract_features=True,
        )  # the parameters match those used to generate the regression data!

        # Load results from previous toolbox version (TF v1 API)
        with open(os.path.join(DATA_PATH, f"{model}_outputs.pkl"), "rb") as f:
            outputs = pickle.load(f)
        taggram_old = outputs["taggram"]
        labels_old = outputs["labels"]
        features_old = outputs["features"]

        # Check results match
        if not np.all(labels == labels_old):
            raise ValueError(f"Labels mismatch for model '{model}'.")

        if not np.allclose(taggram, taggram_old):
            raise ValueError(f"Taggram mismatch for model '{model}'.")

        for key, old_val in features_old.items():
            if not np.allclose(features[key], old_val):
                raise ValueError(
                    f"Features mismatch for model '{model}', feature '{key}'."
                )

        print(f"    Model '{model}' passed the regression test.")

    print("All tests passed successfully.")


if __name__ == "__main__":
    test_regression_extractor()
