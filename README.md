# [**NeurIPS 24**] MetaUAS

[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/csgaobb/MetaUAS)

> Official PyTorch Implementation of [MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning](https://proceedings.neurips.cc/paper_files/paper/2024/file/463a91da3c832bd28912cd0d1b8d9974-Paper-Conference.pdf), NeurIPS 2024.



## Introduction 
MetaUAS unifies anomaly segmentation into change segmentation and provides a pure visual foundation model, which requires only one normal image prompt and no additional training, and effectively and efficiently segments any visual anomalies.  MetaUAS significantly outperforms most zero-shot, few-shot, and even full-shot anomaly segmentation methods. 

## MetaUAS Framework
![overview](images/MetaUAS_Framework.jpg)


## Main Results

![main-results](images/MetaUAS-Main-Results.jpg)

![main-com-eff](images/MetaUAS-Com-Eff.jpg)

![main-diffprompts](images/MetaUASwithDiffprompts.jpg)

![main-diffprompts](images/MetaUAS-Vis.jpg)

## Demo
You can use our [Online Demo](https://huggingface.co/spaces/csgaobb/MetaUAS) to test your custom data for a quick start. Note that the online demo is currently based on CPU.
You could also deploy the demo application to your local CPU/GPU server using the following command:
```
pip install -r requirements.txt
python app.py
```

## Evaluation on MVTec/VisA/GoodsAD
```
bash test.sh
```
## ToDo List
- [x] release pre-trained [MetaUAS models](https://huggingface.co/csgaobb/MetaUAS)
- [x] deploy [online MetaUAS Demo](https://huggingface.co/spaces/csgaobb/MetaUAS)
- [x] open testing code
- [ ] open training code




## Citing
If you find this code useful in your research, please consider citing us:

```
@inproceedings{gao2024metauas,
  title  = {MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning},
  author = {Gao, Bin-Bin},
  booktitle = {Advances in Neural Information Processing Systems},
  pages = {39812--39836},
  year = {2024}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=gaobb/MetaUAS&type=Timeline)](https://www.star-history.com/#gaobb/MetaUAS&Timeline)

## Pair-mask training loss

`train/train_pair_masks.py` uses BCE by default. To try pixel-wise Focal Loss,
add these options to your existing training command:

```bash
python train/train_pair_masks.py <existing-data-and-training-options> --loss-type focal --focal-gamma 2
```

Focal Loss consumes the model's sigmoid probabilities. Training and validation
use the same loss. `--focal-alpha` is optional (disabled by default); when set,
it weights defect pixels by alpha and background pixels by 1-alpha.
`--bce-weight` scales either selected loss for compatibility. Dice loss remains
disabled, while Dice and IoU metrics and best-Dice checkpoints remain available.
The `train/train_pair_masks_bce_only.py` entry point always forces BCE.

Compare BCE and Focal runs using the same data split and training settings, with
separate experiment/checkpoint paths. Compare validation Dice/IoU and error
patterns rather than the numerical loss values across different loss types.

### Device K-fold cross-validation

Run these commands from the repository root in Bash. The existing dataset/GPU
settings live in `scripts_k_fold/train_hedian.sh`. There are 12 device folds;
`train_k_fold/build_hedian_device_manifest.py` only builds their data splits.
All folds train with the shared `train/train_pair_masks.py` loss implementation.

```bash
# Train and test only fold 0 with Focal Loss.
LOSS_TYPE=focal FOCAL_GAMMA=2 bash scripts_k_fold/run_device_folds.sh 0

# Train and test all 12 folds with Focal Loss.
LOSS_TYPE=focal FOCAL_GAMMA=2 bash scripts_k_fold/run_device_folds.sh

# Train and test selected folds, with optional defect-pixel weighting.
LOSS_TYPE=focal FOCAL_GAMMA=2 FOCAL_ALPHA=0.75 bash scripts_k_fold/run_device_folds.sh 0 3 7

# Train only fold 0, without the subsequent test step.
LOSS_TYPE=focal FOCAL_GAMMA=2 bash scripts_k_fold/train_hedian.sh "$(date +%Y%m%d_%H%M%S)" 0

# BCE baseline (also the default when LOSS_TYPE is unset).
LOSS_TYPE=bce bash scripts_k_fold/run_device_folds.sh 0
```

`FOCAL_GAMMA` defaults to 2.0. Leave `FOCAL_ALPHA` unset or empty to disable
class weighting. `BCE_WEIGHT` defaults to 1.0 and scales either loss. These
environment settings are inherited through the multi-fold and train/test wrappers.
The effective loss options are saved in each experiment's `config.json` and
training-state checkpoint. Test inference uses the same probability output and
does not require a loss option. Use distinct run timestamps for separate runs.
