"""
Run experiments in parallel across multiple GPUs.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from itertools import product
from multiprocessing import Pool
from config_dicts import configs


def calculate_runner_splits(total_runners, hparams):
    # Calculate the length of each split
    total_len = len(hparams)
    basic_split = total_len // total_runners
    remainder = total_len % total_runners
    # Calculate the start and end indices of the split
    splits = []
    start = 0
    for runner_index in range(total_runners):
        end = start + basic_split + (1 if runner_index < remainder else 0)
        splits.append(hparams[start:end])
        start = end
    return splits


class HyperParameters:
    def __init__(self, method) -> None:
        self.hparams = method["hparams"]
        self.method = self.hparams["method"]

    def get_product(self):
        # Filter out 'method' from hparams and get the Cartesian product
        filtered_hparams = {key: value for key, value in self.hparams.items() if key != "method"}
        product_list = list(product(*filtered_hparams.values()))

        # Add the 'method' to each combination as a dictionary
        combined_product = [
            {**dict(zip(filtered_hparams.keys(), params)), "method": method_value}
            for method_value in (self.method if isinstance(self.method, list) else [self.method])
            for params in product_list
        ]

        return combined_product


def run_experiment(args):
    hparam, gpu_id, stagger_time = args
    method_name = hparam["method"]
    override = configs[method_name]["base"]
    for key, value in hparam.items():
        if key != "method":
            override[key] = value

    print("override", override)
    time.sleep(stagger_time)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(override, temp_file)
        temp_file_path = temp_file.name
    current_dir = os.getcwd()
    cmd = f"python {os.path.join(current_dir, 'large_legislative_models/main.py')} --config {temp_file_path}"
    print("Running", cmd, file=sys.stderr)
    max_retries = 1
    for attempt in range(max_retries):
        try:
            result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed with error:", e, file=sys.stderr)
            print(e.stdout)
            print(e.stderr, file=sys.stderr)
            if "WandB pipe break" in e.stderr or "WandB pipe break" in e.stdout:
                if attempt < max_retries - 1:
                    print(
                        f"WandB pipe break detected. Retrying in 5 seconds...",
                        file=sys.stderr,
                    )
                    time.sleep(5)
                else:
                    print(
                        f"Max retries reached. Moving to next experiment.",
                        file=sys.stderr,
                    )
            else:
                print(f"Unhandled error. Moving to next experiment.", file=sys.stderr)
    os.unlink(temp_file_path)


def run_batch(batch, gpus, max_stagger_time):
    gpu_assignments = [i % gpus for i in range(len(batch))]
    stagger_times = [random.uniform(0, max_stagger_time) for _ in range(len(batch))]
    experiments_with_gpus_and_stagger = list(zip(batch, gpu_assignments, stagger_times))

    for hparams, gpu_id, stagger_time in experiments_with_gpus_and_stagger:
        print(f"GPU {gpu_id}: {hparams}, {stagger_time}")

    with Pool(processes=len(batch)) as pool:
        pool.map(run_experiment, experiments_with_gpus_and_stagger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MARL experiments in batches with multiple GPUs")
    parser.add_argument("--start", type=int, default=0, help="Start index of experiments")
    parser.add_argument("--end", type=int, default=None, help="End index of experiments (exclusive)")
    parser.add_argument("--num_simultaneous", type=int, default=10, help="Number of experiments to run simultaneously")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--max_stagger_time", type=float, default=20, help="Maximum stagger time in seconds")
    parser.add_argument("--job", type=int, required=True, help="Which job am I?")
    parser.add_argument(
        "--total",
        type=int,
        required=True,
        help="How many total jobs are there running?",
    )

    args = parser.parse_args()

    """
    CHOOSE YOUR CONFIGS HERE
    CHOOSE ASSOCIATED HPARAMS IN config_dicts.py
    """

    methods = ["llm_cleanup"]
    hparams = []

    """
    Calculate product of hparams for each method
    Each set of hparams also includes the method name,
    which is used to select the correct base config in config_dicts.py
    """
    for method in methods:
        method_hparams = HyperParameters(configs[method]).get_product()
        hparams += method_hparams

    """Pick up where another gridsearch left off if it fails"""
    total_experiments = len(hparams)
    start_index = args.start
    end_index = args.end if args.end is not None else total_experiments

    indexed_hparams = hparams[start_index:end_index]
    hparam_splits = calculate_runner_splits(args.total, indexed_hparams)
    job_hparams = hparam_splits[args.job]
    # random.shuffle(job_hparams)
    print(
        "\n\n\n",
        "TOTAL HPARAM SETS:",
        len(indexed_hparams),
        "\n",
        "TOTAL JOB HPARAMS:",
        len(job_hparams),
        "\n\n\n",
        file=sys.stderr,
    )
    print(job_hparams)
    num_batches = math.ceil(len(job_hparams) / args.num_simultaneous)
    print(
        f"Running {len(job_hparams)} experiments in {num_batches} batches of {args.num_simultaneous} experiments each",
        file=sys.stderr,
    )
    print(f"Job hparams: {job_hparams}")
    for i in range(num_batches):
        batch_start = i * args.num_simultaneous
        batch_end = min((i + 1) * args.num_simultaneous, len(job_hparams))
        batch = job_hparams[batch_start:batch_end]
        print(f"\nStarting batch {i+1} of {num_batches}", file=sys.stderr)
        run_batch(batch, args.gpus, args.max_stagger_time)
        print(f"Completed batch {i+1} of {num_batches}", file=sys.stderr)

    print("All experiments completed.", file=sys.stderr)
