#!/bin/bash

# # test
python scripts/eval.py dataset=eval_example sampler=ctrm sampler.num_samples=25
python scripts/eval.py dataset=eval_example sampler=ctrm sampler.num_samples=50
python scripts/eval.py dataset=eval_example sampler=ctrm sampler.num_samples=100
python scripts/eval.py dataset=eval_example sampler=random_shared sampler.num_samples=3000
python scripts/eval.py dataset=eval_example sampler=random_shared sampler.num_samples=5000
python scripts/eval.py dataset=eval_example sampler=random_shared sampler.num_samples=7000
python scripts/eval.py dataset=eval_example sampler=grid_shared sampler.num_samples=1200
python scripts/eval.py dataset=eval_example sampler=grid_shared sampler.num_samples=4500
python scripts/eval.py dataset=eval_example sampler=grid_shared sampler.num_samples=6500
# 
# for dataset in eval_standard eval_wo_obs eval_many_obs eval_more_agents eval_hetero
# do
# for num_samples in 25 50 75 100
# do
# echo "dataset: $dataset | num_samples: $num_samples"
# python scripts/eval.py dataset=$dataset sampler=ctrm sampler.num_samples=$num_samples
# done
# done
# 
# for dataset in eval_standard eval_wo_obs eval_many_obs eval_more_agents 
# do
# for num_samples in 3000 5000 7000
# do
# echo "dataset: $dataset | num_samples: $num_samples"
# python scripts/eval.py dataset=$dataset sampler=random_shared sampler.num_samples=$num_samples
# done
# done

for dataset in eval_standard eval_wo_obs eval_many_obs eval_more_agents 
do
for num_samples in 1200 4500 6500
do
echo "dataset: $dataset | num_samples: $num_samples"
python scripts/eval.py dataset=$dataset sampler=grid_shared sampler.num_samples=$num_samples
done
done

# for dataset in eval_hetero
# do
# for num_samples in 3000 5000 7000
# do
# echo "dataset: $dataset | num_samples: $num_samples"
# python scripts/eval.py dataset=$dataset sampler=random sampler.num_samples=$num_samples
# done
# done

for dataset in eval_hetero
do
for num_samples in 1200 4500 6500
do
echo "dataset: $dataset | num_samples: $num_samples"
python scripts/eval.py dataset=$dataset sampler=grid sampler.num_samples=$num_samples
done
done