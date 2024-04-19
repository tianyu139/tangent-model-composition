# Tangent Model Composition (ICCV 2023, ICLR 2024)
![TMC](tmc.png)

Official code repository for 
- [Tangent Model Composition for Ensembling and Continual Fine-tuning (ICCV 2023)](https://arxiv.org/abs/2307.08114) 
- [Tangent Transformers for Composition, Privacy and Removal (ICLR 2024)](https://arxiv.org/abs/2307.08122)

### Requirements
Our repository is based on PyTorch. We use Torch 1.12 and Python 3.9, other versions have not been tested.

In addition, the following packages are also needed:
```
pip install hydra-core==1.2.0
```

### Datasets
Create a folder for storing datasets in the main directory 
```
mkdir data
```
We provide example scripts for setting up MIT-67 and Oxford Pets in the `setup` directory
```
bash setup/setup_mit.sh
bash setup/setup_oxfordpets.sh
```

### Reproducing results
Our results for the Class Incremental (Class-IL) setting and Data Incremental (Data-IL) can be 
reproduced using 
```
bash scripts/compose.sh`
```
and changing the variables appropriately.



If you find this useful for your work, please consider citing
```
@inproceedings{liu2023tangent,
  title={Tangent Model Composition for Ensembling and Continual Fine-tuning},
  author={Liu, Tian Yu and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={18676--18686},
  year={2023}
}

@article{liu2024tangent,
  title={Tangent transformers for composition, privacy and removal},
  author={Liu, Tian Yu and Golatkar, Aditya and Soatto, Stefano},
  journal={arXiv preprint arXiv:2307.08122},
  year={2024}
}
```


