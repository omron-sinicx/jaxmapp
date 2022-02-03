#!/bin/bash

# test
python scripts/eval.py dataset=eval_example sampler=ctrm sampler.num_samples=50
python scripts/eval.py dataset=eval_example sampler=random_shared sampler.num_samples=3000
python scripts/eval.py dataset=eval_example sampler=grid_shared sampler.num_samples=33

# for dataset in eval_standard eval_wo_obs eval_many_obs eval_more_agents eval_hetero
# do
# for num_samples in 25 50 75 100
# do
# echo "dataset: $dataset | num_samples: $num_samples"
# python scripts/eval.py dataset=$dataset sampler=ctrm sampler.num_samples=$num_samples
# done
# done

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
for num_samples in 1089 4096 7056
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
for num_samples in 1089 4096 7056
do
echo "dataset: $dataset | num_samples: $num_samples"
python scripts/eval.py dataset=$dataset sampler=grid sampler.num_samples=$num_samples
done
done