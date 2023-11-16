#!/usr/bin/python

import os
import functools
import argparse
from pathlib import Path
from absl import app
from absl import logging
from absl.flags import argparse_flags

import ray
import pandas as pd
from torch.utils.data import DataLoader

from models import Trainer, Evaluator
from datasets import ImageNet1K, transform_fn
from utils import set_seed, parse_target, get_feasible_qualities


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--gpu", "-g", type=int, default=0,
        help="GPU index to use.")
    parser.add_argument(
        "--quality", "-q", type=int, default=1,
        help="Quality of pre-trained surrogate model used for training.")
    parser.add_argument(
        "--lambda",  "-ld", type=float, dest="lmbda", default=0.1,
        help="Lagrangian multiplier for RD tradeoffs.")
    parser.add_argument(
        "--target", "-t", type=str, default='resnet18',
        help="Target distortion (ms-ssim, resnet18, inception_v3).")
    parser.add_argument(
        "--steps", "-s", type=int, default=100000,
        help="Total learning steps.")
    parser.add_argument(
        "--optimizer", "-opt", type=str, default='adam',
        help="Optimizer (adam, sgd).")
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=0.0001,
        help="Learning rate.")
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=64,
        help="Batch size.")
    parser.add_argument(
        "--checkpoint_period", "-cp", type=int, default=10000,
        help="Number of steps to save a checkpoint.")
    parser.add_argument(
        "--temp_checkpoint_period", "-tcp", type=int, default=1000,
        help="Number of steps to save a checkpoint temporarily.")
    parser.add_argument(
        "--num_parallel_load", "-npl", type=int, default=8,
        help="Number of parallel loaders.")
    parser.add_argument(
        "--num_parallel_eval", "-npe", type=int, default=3,
        help="Number of parallel evaluators.")
    parser.add_argument(
        "--eval_codec", "-ec", type=str, default='jpeg',
        help="Codec being used in evaluation (jpeg, webp, surr).")
    parser.add_argument(
        "--eval_target", "-et", type=str, default='resnet18',
        help="Target distortion (msssim, resnet18, inception_v3).")
    parser.add_argument(
        "--eval_period", "-ep", type=int, default=50000,
        help="Number of steps to evaluate periodically.")
    parser.add_argument(
        "--seed", type=int,
        help="Random seed.")
    parser.add_argument(
        "--output", "-o", type=str, default="out",
        help="Name of output directory.")
    args = parser.parse_args(argv[1:])
    return args


def main(args):
    assert args.eval_period % args.checkpoint_period == 0, "'--eval_period' has to divided by '--checkpoint_period'"

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    set_seed(args.seed)
    session_name = (f'{args.target}_q{args.quality}_ld{args.lmbda}_bs{args.batch_size}_'
                    f'{args.optimizer}_lr{args.learning_rate}')
    logging.info(f"Start a training process for '{session_name}'.")

    resizing_size = parse_target(args.target)['resizing_size']
    transform = functools.partial(transform_fn, resizing_size=resizing_size)
    train_ds = ImageNet1K('/data/imagenet/', 'train', transform=transform)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          num_workers=args.num_parallel_load, pin_memory=False)
    train_it = iter(train_dl)
    
    valid_ds = ImageNet1K('/data/imagenet/', 'val', transform=transform)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                          num_workers=args.num_parallel_load, pin_memory=False)

    trainer_builder = ray.remote(num_gpus=1)(Trainer)
    evaluator_builder = ray.remote(num_gpus=1/args.num_parallel_eval)(Evaluator)

    # Start ray driver & create a trainer.
    ray.init()
    trainer = trainer_builder.remote(args, session_name=session_name)
    step = ray.get(trainer.get_step.remote())
    
    # Training
    while True:
        if step >= args.steps:
            logging.info(f"Training is finished at {step} steps.")
            break

        # Feed input data to the `trainer``.
        try:
            data = next(train_it)
        except StopIteration:
            train_it = iter(train_dl)
            data = next(train_it)

        # Run background training loop.
        ray.get(trainer.put_data.remote(data))

        # Check training progress by `step`.
        step = ray.get(trainer.get_step.remote()) 

        # Check for evaluation
        if step and step % args.eval_period == 0:

            checkpoint_path = Path(args.output) / session_name

            # Delete unused intermediate checkpoints.
            unused_ckpts = filter(lambda x: int(x.name.split('_')[-1][:-4]) % args.checkpoint_period,
                                  checkpoint_path.glob('*.pth'))
            [ckpt.unlink() for ckpt in unused_ckpts]

            csv_prefix = f"{args.eval_codec}_{args.eval_target}"
            eval_path = checkpoint_path / f'{csv_prefix}_{step}.csv'
            qualities = get_feasible_qualities(args.eval_codec)

            if os.path.exists(eval_path):
                df = pd.read_csv(eval_path)
                qualities -= set(df.quality)
                if not qualities:
                    # if all `quality` is evaluated,
                    # go to next training step.
                    continue

            # Start evaluation.

            # Kill trainer.
            ray.kill(trainer)

            evaluators = [evaluator_builder.remote(args, checkpoint_path) for _ in range(args.num_parallel_eval)]
            ray.get([evaluator.load_weight.remote(-1) for evaluator in evaluators])
            runners = []
            while qualities:
                while len(qualities) and len(evaluators):
                    evaluator = evaluators.pop(0)
                    quality = qualities.pop()
                    ray.get(evaluator.set_quality.remote(quality))
                    runners.append(evaluator)

                # Run background evaluation loop.
                ray.get([evaluator.run.remote() for evaluator in runners])

                # Send data.
                for data in valid_dl:
                    data = ray.put(data)
                    ray.get([evaluator.put_data.remote(data) for evaluator in runners])

                # Send end signal.
                ray.get([evaluator.put_data.remote(None) for evaluator in runners])

                # Wait for outputs. Process and save output data.
                outputs = ray.get([evaluator.get_output.remote() for evaluator in runners])
                metric_arr, bpp_arr, quality_arr = zip(*outputs)
                df = pd.DataFrame({args.target: metric_arr, 'bpp': bpp_arr, 'quality': quality_arr})
                df.to_csv(eval_path, mode='a', index=False, header=(not os.path.exists(eval_path)))

                # Prepare next evaluation loop.
                evaluators.extend(runners)
                runners.clear()
            
            # Kill evaluators.
            [ray.kill(evaluator) for evaluator in evaluators]

            # Create a trainer.
            trainer = trainer_builder.remote(args, session_name=session_name)
            step = ray.get(trainer.get_step.remote())


if __name__ == '__main__':
    try:
        app.run(main, flags_parser=parse_args)
    except KeyboardInterrupt:
        print("Interrupted..")
        ray.shutdown()
        exit(1)