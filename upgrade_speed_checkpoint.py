"""Upgrade a legacy m/s HighwayGRL checkpoint to canonical km/h decoding.

Because the average-speed target is z-scored, multiplying both scaler mean and
standard deviation by 3.6 produces the same normalized target and model
weights as retraining on uniformly converted km/h targets.
"""

import argparse
import os
import shutil
import tempfile

import torch


def upgrade_checkpoint(path, *, apply=False, factor=3.6):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    meta = checkpoint.setdefault("meta", {})
    scalers = checkpoint.get("scalers", {})
    if "avg_mu" not in scalers or "avg_sd" not in scalers:
        raise ValueError("Checkpoint does not contain avg-speed scaler values")
    if meta.get("avg_speed_units") == "km/h":
        print("Checkpoint is already marked as canonical km/h; no change needed.")
        return

    old_mu = float(scalers["avg_mu"])
    old_sd = float(scalers["avg_sd"])
    new_mu = old_mu * factor
    new_sd = old_sd * factor
    print(f"avg scaler mean: {old_mu:.6f} -> {new_mu:.6f} km/h")
    print(f"avg scaler std:  {old_sd:.6f} -> {new_sd:.6f} km/h")

    if not apply:
        print("Audit only: checkpoint was not changed. Pass --apply to upgrade it.")
        return

    backup = path + ".legacy-mps.bak"
    if os.path.exists(backup):
        raise FileExistsError(f"Backup already exists: {backup}")
    shutil.copy2(path, backup)

    scalers["avg_mu"] = new_mu
    scalers["avg_sd"] = new_sd
    meta["avg_speed_units"] = "km/h"
    meta["source_avg_speed_units"] = "m/s converted to km/h"
    meta["speed_unit_upgrade_factor"] = factor

    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".speed-checkpoint-", suffix=".pt", dir=directory
    )
    os.close(descriptor)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    print(f"Upgraded checkpoint: {path}")
    print(f"Recoverable backup:   {backup}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--factor", type=float, default=3.6)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    upgrade_checkpoint(args.checkpoint, apply=args.apply, factor=args.factor)


if __name__ == "__main__":
    main()
