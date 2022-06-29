# advPatch-pytorch
code for generating adversarial patches

# Required packages
pip install pytorch-msssim


# Installation

Clone the project with the submodules.

```bash
git clone --rescursive URL
```

This repo requires Python >= 3.6.
To get dependent packages, you can install the required packages in `requirement.txt` via
```bash
pip install -r requirement.txt
```

In order to use the object detectors from `SSD` or `Detectrons`, you will need to install them from the submodule

To install SSD, more details can be found [here](https://github.com/lufficc/SSD#installation).

```bash
cd REPO/detector/SSD
pip install -e . 
``` 

To install Detectron2, more details can be found [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

```bash
cd REPO/detector/detectron2
pip install -e .
``` 




#Usage

1 Learning STN with Generator
1) Training

```bash
python train_patchTransformer.py --config configs/config_patchTransformer.yaml --logdir STN-results --dataset neu_color \
   --datadir ../../adv_data/neu_data --epochs 600 --STN tps --learnableSTN --use_LCT --LightingCT gen --batch_size 72`
```
2) Evaluation

```bash
python train_patchTransformer.py --config configs/config_patchTransformer.yaml --logdir thinklab-STN-results --dataset neu_color \
  --datadir ../../adv_data/neu_data --epochs 600 --STN tps --learnableSTN --use_LCT --LightingCT gen --batch_size 60 \
  --patch_transformer_path thinklab-STN-results/PT_neu_color_STN_resnet18_ds128_fc256_tps_bounded20x10_gen_p256_L1Mask_bs60_e600/model_best.pth.tar \
  --visualize --evaluate
```

--visualize: save intermediate results into a folder 'vis_output' under 'patch_transformer_path'
--val_list_file: specify which subset to be evaluated: train or validation

2 Learning Printer Color (PCT) and Lighting Transformation (LCT)
1) Training

```bash
python train_patchTransformer.py --config configs/config_patchTransformer.yaml --logdir PatNet-results --dataset neu_color \
--datadir ../../adv_data/neu_data --epochs 600 --STN tps --use_PCT --PrinterCT PCTLinear --use_LCT --LightingCT cc --batch_size 72 \
--pretrained STN-results/PT_neu_color_STN_resnet18_ds128_fc256_tps_bounded20x10_gen_bs72_e600/model_best.pth.tar 
```
3 Learning Adversarial Attack Model
1) Training 

```bash
python train_advPatch.py --config configs/config_advPatch.yaml --logdir AdvNet-results --dataset neu_color --datadir ../../adv_data/neu_data \
--epochs 600 --STN tps --use_PCT --PrinterCT PCTLinear --use_LCT --LightingCT cc --batch_size 72  \
--patch_transformer_path PatNet-results/PatNet/PT_neu_color_fixedSTN_blur6_resnet18_ds128_fc256_tps_bounded20x10_PCTLinear_cc_alexnet_bs72_e600_pretrained_nopctloss_blur/model_best.pth.tar
```

2) Evaluation

```bash
python train_advPatch.py --config configs/config_advPatch.yaml --logdir AdvNet-results --dataset neu_color --datadir ../../adv_data/neu_data \
--epochs 600 --STN tps --use_PCT --PrinterCT PCTLinear --use_LCT --LightingCT cc --batch_size 72  \
--patch_transformer_path PatNet-results/PatNet/PT_neu_color_fixedSTN_blur6_resnet18_ds128_fc256_tps_bounded20x10_PCTLinear_cc_alexnet_bs72_e600_pretrained_nopctloss_blur/model_best.pth.tar\
--evaluate 
```
