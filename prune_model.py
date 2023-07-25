import os
import shutil


def prune_model(
    model_path: str,
    snapshot: int = None,
    interation: int = None,
    shuffle=None,
    out_dir: str = None,
):
    if out_dir is None:
        out_dir = model_path + "_pruned"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if interation is None:
        pass
    if snapshot is None:
        f"/iteration-{interation}/"
        f"dlc-models/iteration-{interation}/"
