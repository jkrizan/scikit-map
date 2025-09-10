#!/bin/bash
python cfc_train_v6.py lightning_logs/version_15/checkpoints/cfc_v6_b2_epoch-093.ckpt
python cfc_train_v6.py lightning_logs/version_16/checkpoints/cfc_v6_b3_epoch-089.ckpt
python cfc_train_v6.py lightning_logs/version_18/checkpoints/cfc_v6_b5_epoch-095.ckpt