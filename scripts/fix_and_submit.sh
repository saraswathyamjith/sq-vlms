#!/bin/bash
module load miniforge
source activate sq-vlm
pip install "datasets>=3.0,<4"
cd ~/self-questioning-vlms
git pull
sbatch scripts/train.sh
