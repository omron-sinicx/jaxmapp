#!/bin/bash

# test
python scripts/eval.py dataset=eval_example sampler=ctrm sampler.num_samples=25
python scripts/eval.py dataset=eval_example sampler=ctrm sampler.num_samples=50
python scripts/eval.py dataset=eval_example sampler=ctrm sampler.num_samples=100
python scripts/eval.py dataset=eval_example sampler=random_shared sampler.num_samples=3000
python scripts/eval.py dataset=eval_example sampler=random_shared sampler.num_samples=5000
python scripts/eval.py dataset=eval_example sampler=random_shared sampler.num_samples=7000
python scripts/eval.py dataset=eval_example sampler=grid_shared sampler.num_samples=1200
python scripts/eval.py dataset=eval_example sampler=grid_shared sampler.num_samples=4500
python scripts/eval.py dataset=eval_example sampler=grid_shared sampler.num_samples=6500
