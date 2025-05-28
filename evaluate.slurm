#!/bin/bash
#SBATCH --job-name=gemma3_vistw
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=0-24:00:00
#SBATCH --partition tp1-an1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chenjoachim63@outlook.com

source ~/miniconda3/bin/activate vistw-env

export OAI_KEY=$(cat .secrets/openai_key)

# 執行腳本
echo "Running inference script..."

python -m simplevals.eval google/gemma-3-27b-it \
	--series gemma3 \
	--dataset music natural_science navigation pharmaceutical_chemistry physics sociology statistics structural_engineering veterinary_medicine \
	--mode image

echo "Script completed at: $(date)"

