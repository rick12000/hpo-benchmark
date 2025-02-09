#!/usr/bin/env python
# train_custom.py

import sys
import os
import pickle
from syne_tune import Reporter


def get_config_from_argv():
    """
    Automatically ingests hyperparameters passed as command-line arguments.
    Expected format: --key value
    Returns a dictionary mapping each key to its value (attempting type conversion).
    """
    args = sys.argv[1:]
    config = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            # Check if there is a following value that doesn't start with --
            if (i + 1) < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                i += 2
            else:
                value = True  # Treat flags without a value as True
                i += 1
            # Try to convert to int or float if possible.
            try:
                try:
                    if "true" in value.lower():
                        value = True
                    elif "false" in value.lower():
                        value = False
                except Exception:
                    pass
                if isinstance(value, bool):  # Avoid trying to cast boolean values
                    pass
                elif "." in str(value):
                    value = float(value)
                else:
                    value = int(value)
            except (ValueError, TypeError):
                # Leave value as a string if conversion fails.
                pass
            config[key] = value
        else:
            i += 1
    return config


def main():
    # Load the performance generator from a pickle file.
    pg_pickle = "cache/syne-tune/performance_generator.pkl"
    if not os.path.exists(pg_pickle):
        raise FileNotFoundError(f"Could not find {pg_pickle}.")
    with open(pg_pickle, "rb") as f:
        performance_generator = pickle.load(f)

    # Automatically ingest hyperparameters into a config dictionary.
    config = get_config_from_argv()

    del config["st_checkpoint_dir"]

    # Optionally, print the configuration for debugging:
    print("Configuration received:", config)

    # Use the performance generator to predict a metric based on the config.
    mse = performance_generator.predict(config)
    print(mse)

    # Report the metric to Syne Tune.
    reporter = Reporter()
    reporter(mse=mse)


if __name__ == "__main__":
    main()
