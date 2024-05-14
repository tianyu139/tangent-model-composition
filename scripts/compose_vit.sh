dataset=mit67
n_tasks=50          # number of shards

# 50 Tasks - TAFT-1 on ViT-L dataset
# In the paper, we set augment=False to improve reproducibility and reduce stochasticity in the results.
# You can also consider setting augment=True, which is likely to produce much better results than reported.

python main.py -m name=${dataset}-${n_tasks} arch=vitl16 dataset=${dataset} +preset=tangent trainer=tangent mse_weight=15 n_tasks=${n_tasks} task_split=rand epochs=30 lr=1e-3 tangent_unfreeze_layer=23 replace_cls_token=False val_freq=999 pretrained=False augment=False
