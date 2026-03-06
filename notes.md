Turn on Virtual Env: "source venv/bin/activate"

Remove old checkpoints:
rm -f outputs/checkpoints/*.pth

Baseline CNN (~8 min): [RERUN - old run had no best checkpoint saving]
sudo nice -n -20 python ml/scripts/trains/train_baseline.py --epochs 20 --batch_size 16 --lr 5e-4

U-Net (~45 min): [RERUN - was interrupted]
sudo nice -n -20 python ml/scripts/trains/train_unet.py --epochs 30 --batch_size 8 --lr 1e-4

Pix2Pix GAN (~6 hrs):
sudo nice -n -20 python ml/scripts/trains/train_gan.py --epochs 50 --batch_size 4 --lr 2e-4 --lambda_l1 100

Fusion GAN (~7 hrs):
sudo nice -n -20 python ml/scripts/trains/train_fusion.py --epochs 50 --batch_size 4 --lr 2e-4 --lambda_l1 100

Best checkpoints to use for inference:
  baseline_cnn_best.pth
  unet_best.pth
  gan_generator_best.pth
  fusion_generator_best.pth
