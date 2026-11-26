# AEAFusion

AEAFusion fuses paired visible and infrared images under degraded imaging conditions. The first stage restores the two inputs and predicts a spatial reliability map for each modality. The second stage uses these maps and cross-modal conflict to assign weights to visible-specific, infrared-specific, shared and rejected detail.

## Pretrained inference

The inference script uses PyTorch, torchvision, NumPy and Pillow. Prepare aligned visible and infrared images with matching filename stems and spatial dimensions. See [Data preparation](data/README.md) for the dataset layout.

Set `VIS_LOW_DIR`, `IR_NOISY_DIR` and `OUT_DIR` in [the inference script](fusion-stagev2/test_stage2_cara_full.py) to your local directories. Set `RESTORER_CKPT` and `FUSION_CKPT` to the corresponding checkpoint paths below. Paths are relative to the repository root.

| Imaging condition | Stage 1 restoration | Stage 2 fusion |
| --- | --- | --- |
| Low light | `restoration-stage/fusion_checkpoints/best_model.pth` | `fusion-stagev2/fusion_checkpoints/fusion_cara_full_epoch_0100.pth` |
| Overexposure | `restoration-stage/exposure_checkpoints/best_vis.pth` | `fusion-stagev2/exposure_checkpoints/fusion_cara_full_epoch_0100.pth` |

Run the following command from the repository root. Fused images are written to `OUT_DIR`.

```sh
python fusion-stagev2/test_stage2_cara_full.py
```

## Evaluation

Install `piq` to compute all five fusion metrics. In [the evaluation script](fusion-stagev2/eval_denoif_metrics.py), set `FUSED_DIR` to the output directory and point `VIS_CLEAN_DIR` and `IR_CLEAN_DIR` to the paired clean references. The script gives equal weight to the visible and infrared references for SSIM, PSNR, FSIM and CC. It computes BRISQUE on each fused image.

```sh
python fusion-stagev2/eval_denoif_metrics.py
```

## Training

[Stage 1](restoration-stage/train.py) trains the restoration network with degraded inputs and clean references. Its additional dependencies include `torchmetrics` and `pytorch-msssim`. [Stage 2](fusion-stagev2/train_stage2_cara_full.py) trains the fusion network with a pretrained restoration checkpoint. Set the input, checkpoint and output paths in the training scripts before running them. The [baseline configuration](run-records/hyperparameter-baseline/config.json) lists the Stage 2 settings for the 100-epoch baseline run.
