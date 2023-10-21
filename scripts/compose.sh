dataset=mit67 # choices: oxfordpets, caltech256, mit67
n_tasks=5          # number of shards

# Class Incremental Learning
python main.py -m name=${dataset}-cil dataset=${dataset} +preset=tangent trainer=tangent mse_weight=15 n_tasks=${n_tasks}

# Data Incremental Learning
python main.py -m name=${dataset}-dil dataset=${dataset} +preset=tangent trainer=tangent mse_weight=5 n_tasks=${n_tasks} task_split=rand
