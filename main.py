import os
import logging
import argparse

logging.basicConfig(level=logging.ERROR)
try:
    os.add_dll_directory(os.path.join(os.environ.get("CUDA_PATH_V11_2"), "bin"))
except (AttributeError, TypeError):
    logging.info("cuda not loaded")
from aemotrics import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unilateral facial palsey quantification using DeepLabCut markerless point tracking."
    )
    parser.add_argument(
        "-m",
        "--model-path",
        dest="model_path",
        type=str,
        nargs=1,
        default=None,
        help="The path to the directory containing the deeplabcut model",
    )
    args = parser.parse_args()
    main(args.model_path)
