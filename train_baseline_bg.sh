#!/bin/bash
cd /Users/ihorpanasenko/Documents/University/Diploma/ImageColorizationAnsamble
source venv/bin/activate
nohup python -u ml/scripts/trains/train.py --model baseline --epochs 20 --batch_size 16 --data_path ./data/coco/val2017 --save_dir ./outputs/checkpoints > outputs/train_baseline.log 2>&1 &
echo "Baseline training started with PID: $!"
