#!/usr/bin/env python3
"""Run the pair-mask trainer with BCE loss only.

This is a non-invasive experiment entry point: it reuses train_pair_masks.py
without modifying it, while forcing ``--dice-weight 0`` for both training and
validation. Dice and IoU are still reported as evaluation metrics.
"""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def force_option(argv, option, value):
    """Replace every spelling of an option with one final forced value."""
    filtered = [argv[0]]
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == option:
            index += 2
            continue
        if argument.startswith(f'{option}='):
            index += 1
            continue
        filtered.append(argument)
        index += 1
    filtered.extend([option, str(value)])
    return filtered


def zero_dice_loss(pred, target, eps=1e-6):
    """Return a scalar zero so Dice loss is not built into the graph."""
    del target, eps
    return pred.new_zeros(())


def main():
    sys.argv = force_option(sys.argv, '--dice-weight', 0.0)
    sys.argv = force_option(sys.argv, '--loss-type', 'bce')
    from train import train_pair_masks as trainer

    trainer.dice_loss = zero_dice_loss
    if int(os.environ.get('RANK', '0')) == 0:
        print('[train_pair_masks_bce_only] BCE-only mode: dice_weight=0', flush=True)
    trainer.main()


if __name__ == '__main__':
    main()
