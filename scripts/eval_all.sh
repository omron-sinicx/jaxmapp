#!/bin/bash

for dataset in eval_standard eval_wo_obs eval_many_obs eval_more_agents eval_hetero
do
for num_samples in 25 50 75 100
do
echo "dataset: $dataset | num_samples: $num_samples"
python scripts/eval.py dataset=$dataset sampler=ctrm sampler.num_samples=$num_samples
done
done

for dataset in eval_standard eval_wo_obs eval_many_obs eval_more_agents 
do
for num_samples in 3000 5000 7000
do
echo "dataset: $dataset | num_samples: $num_samples"
python scripts/eval.py dataset=$dataset sampler=random_shared sampler.num_samples=$num_samples
done
done

for dataset in eval_standard eval_wo_obs eval_many_obs eval_more_agents 
do
for num_samples in 1200 4500 6500
do
echo "dataset: $dataset | num_samples: $num_samples"
python scripts/eval.py dataset=$dataset sampler=grid_shared sampler.num_samples=$num_samples
done
done

for dataset in eval_hetero
do
for num_samples in 3000 5000 7000
do
echo "dataset: $dataset | num_samples: $num_samples"
python scripts/eval.py dataset=$dataset sampler=random sampler.num_samples=$num_samples
done
done

for dataset in eval_hetero
do
for num_samples in 1200 4500 6500
do
echo "dataset: $dataset | num_samples: $num_samples"
python scripts/eval.py dataset=$dataset sampler=grid sampler.num_samples=$num_samples
done
done