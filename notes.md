Turn on Virtual Env: "source venv/bin/activate"

Fix checkpoint ownership (files may be owned by root):
sudo chown -R $(whoami) outputs/checkpoints/

Remove old checkpoints:
rm -f outputs/checkpoints/*.pth


## Training Commands (v2 — with improvements)

### Improvements over v1:
- Train/val split (90/10) with validation PSNR-based best checkpoint saving
- VGG perceptual loss for GAN/Fusion (more stable training, better color accuracy)
- Label smoothing (0.1) prevents discriminator overconfidence
- Gradient clipping (max_norm=1.0) prevents generator gradient explosion
- Warm-start GAN/Fusion from pretrained U-Net weights
- Random horizontal flip data augmentation
- Separate G/D learning rates available (--lr_g, --lr_d)

### Stage 1 — Baseline CNN (~16 min):
sudo nice -n -20 python ml/scripts/trains/train_baseline.py --epochs 20 --batch_size 16 --lr 5e-4 --val_ratio 0.1 --val_every 5 --num_samples 2000

### Stage 2 — U-Net (~90 min):
sudo nice -n -20 python ml/scripts/trains/train_unet.py --epochs 30 --batch_size 8 --lr 1e-4 --val_ratio 0.1 --val_every 5 --num_samples 2000

### Stage 3 — Pix2Pix GAN (~4 hrs with 2000 imgs, ~6 hrs with all 5000):  [warm-start from U-Net]
sudo nice -n -20 python ml/scripts/trains/train_gan.py --epochs 50 --batch_size 4 --lr 2e-4 --lambda_l1 100 --lambda_perceptual 10 --label_smoothing 0.1 --warmstart_g outputs/checkpoints/unet_best.pth --val_ratio 0.1 --val_every 5 --num_samples 2000

### Stage 4 — Fusion GAN (~5 hrs with 2000 imgs, ~7 hrs with all 5000):  [warm-start from U-Net, partial weight loading]
sudo nice -n -20 python ml/scripts/trains/train_fusion.py --epochs 50 --batch_size 4 --lr 2e-4 --lambda_l1 100 --lambda_perceptual 10 --label_smoothing 0.1 --warmstart_g outputs/checkpoints/unet_best.pth --val_ratio 0.1 --val_every 5 --num_samples 2000


## Best checkpoints to use for inference:
  baseline_cnn_best.pth
  unet_best.pth
  gan_generator_best.pth       (now saved by validation PSNR, not G loss)
  fusion_generator_best.pth    (now saved by validation PSNR, not G loss)

## Notes:
- Train in order: Baseline → U-Net → GAN → Fusion (GAN/Fusion warm-start from U-Net)
- Best checkpoints are now selected by highest validation PSNR (not lowest loss)
- --num_samples caps the total image pool (2000 = ~8 min/epoch for GAN, 5000 = ~21 min/epoch)
- Omit --num_samples to use all 5000 images for best quality
- See docs/GAN_TRAINING_RESEARCH.md for full analysis of v1 training issues
