# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import unittest

import nlgeval
from nlgeval import NLGEval

from pathlib import Path

def test_compute_metrics(mode='model'):
    # The example from the README.
    root_dir = Path(os.path.dirname(__file__))
    root_dir = root_dir / 'evaluation_folder'
    hypothesis = root_dir / '{}_base_gen.txt'.format(mode)
    references = [root_dir / 'human_base_gen.txt']

    assert hypothesis.exists()
    assert references[0].exists()
    scores = nlgeval.compute_metrics(hypothesis, references)
    return

def sample_test_compute_metrics():
    # The example from the README.
    root_dir = os.path.join(os.path.dirname(__file__), 'evaluation_folder')
    hypothesis = os.path.join(root_dir, 'hyp.txt')
    references = [os.path.join(root_dir, 'ref1.txt')]
    scores = nlgeval.compute_metrics(hypothesis, references)
    return


if __name__== "__main__":
    test_compute_metrics(mode='naive')
