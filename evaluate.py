#!/usr/bin/python

import os
import math
import itertools
import functools
import argparse
from collections import defaultdict
from pathlib import Path
from absl import app
from absl import logging
from absl.flags import argparse_flags

import ray
import pandas as pd
from torch.utils.data import DataLoader

from models import Evaluator
from datasets import ImageNet1K, transform_fn
from utils import set_seed, parse_target, get_feasible_qualities


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--gpu", "-g", type=int, default=0,
        help="GPU index to use.")
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=64,
        help="Batch size.")
    parser.add_argument(
        "--num_parallel_load", "-npl", type=int, default=8,
        help="Number of parallel loaders.")
    parser.add_argument(
        "--checkpoint_path", "-cp", type=str,
        help="Saved checkpoint path")
    parser.add_argument(
        "--num_parallel_eval", "-npe", type=int, default=3,
        help="Number of parallel evaluators.")
    parser.add_argument(
        "--eval_quality", "-eq", type=str, default="1-10",
        help="qualities to evaluate (comma-separated or dashed).")
    parser.add_argument(
        "--eval_codec", "-ec", type=str, default='jpeg',
        help="Codec being used in evaluation (jpeg, webp, vtm, surr).")
    parser.add_argument(
        "--eval_target", "-et", type=str, default='resnet18',
        help="Target distortion (msssim, resnet18, inception_v3).")
    parser.add_argument(
        "--eval_period", "-ep", type=int, default=50000,
        help="Number of steps to evaluate periodically.")
    parser.add_argument(
        "--num_eval_data", "-ned", type=int, default=-1,
        help="Number of data used in evaluation (-1 means all).")
    parser.add_argument(
        "--seed", type=int,
        help="Random seed.")
    args = parser.parse_args(argv[1:])
    return args


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    set_seed(args.seed)
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Start a evaluation process for '{checkpoint_path.name}'.")

    resizing_size = parse_target(args.eval_target)['resizing_size']
    transform = functools.partial(transform_fn, resizing_size=resizing_size)
    
    valid_ds = ImageNet1K('/data/imagenet/', 'val', transform=transform)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                          num_workers=args.num_parallel_load, pin_memory=False)

    evaluator_builder = ray.remote(num_gpus=1/args.num_parallel_eval)(Evaluator)

    if '-' in args.eval_quality:
        s, e = map(int, args.eval_quality.split('-'))
        eval_qualities = set(range(s, e+1))
    elif ',' in args.eval_quality:
        eval_qualities = set(map(int, args.eval_quality.split(',')))
    else:
        eval_qualities = set(int(args.eval_quality))

    # Limit qualities.
    qualities = get_feasible_qualities(args.eval_codec)
    
    eval_qualities &= set(qualities)
    eval_qualities = sorted(eval_qualities)

    # CSV file prefix.
    csv_prefix = f"{args.eval_codec}_{args.eval_target}"

    # Start ray driver
    ray.init()

    # Evaluation
    evaluators = []
    while True:

        # Make all possible candidates.
        if 'base' in checkpoint_path.name:
            eval_ckpt_steps = ['base']
        else:
            eval_ckpts = checkpoint_path.glob('*.pth')
            eval_ckpt_steps = map(lambda x: int(x.name.split('_')[-1][:-4]), eval_ckpts)
            eval_ckpt_steps = sorted(filter(lambda x: x % args.eval_period == 0, eval_ckpt_steps))
        eval_cfgs = set(itertools.product(eval_ckpt_steps, eval_qualities))
        
        # Find existing configs and delete them from the candidates.    
        existing_csvs = checkpoint_path.glob(f'{csv_prefix}_*.csv')
        for csv_file in existing_csvs:
            if 'base' in csv_file.name:
                step = 'base'
            else:
                step = int(csv_file.name.split('_')[-1][:-4])
            qs = set(pd.read_csv(csv_file).quality)
            cfgs = set(itertools.product([step], qs))
            eval_cfgs -= cfgs
        
        eval_cfgs = sorted(eval_cfgs, reverse=True)

        if not eval_cfgs:
            logging.info("All evaluation is finished.")
            break

        curr_cfgs = defaultdict(list)
        for step, quality in eval_cfgs[:args.num_parallel_eval]:
            curr_cfgs[step].append(quality)

        # Re-use created evaluators.
        if evaluators:
            ready, not_ready = [], []
            while evaluators:
                evaluator = evaluators.pop(0)
                step = ray.get(evaluator.get_step.remote())
                if step in curr_cfgs.keys():
                    quality = curr_cfgs[step].pop(0)
                    ray.get(evaluator.set_quality.remote(quality))
                    if not curr_cfgs[step]:
                        curr_cfgs.pop(step)
                    ready.append(evaluator)
                else:
                    not_ready.append(evaluator)

            curr_cfgs = [(s, q) for s, qs in curr_cfgs.items() for q in qs]
            curr_cfgs = sorted(curr_cfgs, reverse=True)
            if not_ready:
                ray.get([e.load_weight.remote(s) for e, (s, _) in zip(not_ready, curr_cfgs)])
                ray.get([e.set_quality.remote(q) for e, (_, q) in zip(not_ready, curr_cfgs)])
                ready.extend(not_ready)
            evaluators = ready

        # Create new evaluators (occur during first evaluation).
        else:
            curr_cfgs = [(s, q) for s, qs in curr_cfgs.items() for q in qs]
            evaluators = [evaluator_builder.remote(args, checkpoint_path) for _ in range(args.num_parallel_eval)]
            ray.get([e.load_weight.remote(s) for e, (s, _) in zip(evaluators, curr_cfgs)])
            ray.get([e.set_quality.remote(q) for e, (_, q) in zip(evaluators, curr_cfgs)])

        # Run background evaluation loop.
        ray.get([evaluator.run.remote() for evaluator in evaluators])

        # Send data.
        cnt = 0
        valid_it = iter(valid_dl)
        while True:
            try:
                data = next(valid_it)
            except StopIteration:
                break

            data = ray.put(data)
            ray.get([evaluator.put_data.remote(data) for evaluator in evaluators])
            cnt += 1
            if cnt == math.ceil(args.num_eval_data / args.batch_size):
                break

        # Send end signal.
        ray.get([evaluator.put_data.remote(None) for evaluator in evaluators])

        # Wait for outputs. Process and save output data.
        outputs = ray.get([evaluator.get_output.remote() for evaluator in evaluators])
        for (metric, bpp, quality), evaluator in zip(outputs, evaluators):
            step = ray.get(evaluator.get_step.remote())
            # if args.num_eval_data == -1:
            #     eval_path = checkpoint_path / f'{csv_prefix}_{step}.csv'
            # else:
            #     eval_path = checkpoint_path / f'{csv_prefix}_{step}_n{args.num_eval_data}.csv'
            eval_path = checkpoint_path / f'{csv_prefix}_{step}.csv'
            df = pd.DataFrame({args.eval_target: [metric], 'bpp': [bpp], 'quality': [quality]})
            df.to_csv(eval_path, mode='a', index=False, header=(not os.path.exists(eval_path)))


if __name__ == '__main__':
    try:
        app.run(main, flags_parser=parse_args)
    except KeyboardInterrupt:
        print("Interrupted..")
        ray.shutdown()
        exit(1)