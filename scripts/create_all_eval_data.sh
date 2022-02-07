#!/bin/bash

## data creation
python scripts/create_eval_data.py dataset=eval_standard
python scripts/create_eval_data.py dataset=eval_wo_obs
python scripts/create_eval_data.py dataset=eval_many_obs
python scripts/create_eval_data.py dataset=eval_more_agents
python scripts/create_eval_data.py dataset=eval_hetero
