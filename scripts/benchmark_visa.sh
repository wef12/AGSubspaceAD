#!/bin/bash
#SBATCH --partition=gpu_h100     # GPU partition
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --job-name=visa-few-shot # job name
#SBATCH --ntasks=1               # number of tasks
#SBATCH --cpus-per-task=4        # number of CPU cores
#SBATCH --mem=128G               # memory per node
#SBATCH --time=24:00:00          # walltime
#SBATCH --output=logs/out_%j.txt # standard output
#SBATCH --error=logs/err_%j.txt  # standard error


mkdir -p logs

eval "$($CONDA_EXE shell.bash hook)"
conda activate subspacead

# Set the absolute path to your VisA dataset directory
VISA_PATH="/workspace/datasets/VisA_pytorch"

# Categories to run, space-separated (e.g. "candle cashew pcb1").
# Leave empty to run all categories found in VISA_PATH.
CATEGORIES="pcb2"

# Number of normal reference samples per category, space-separated (e.g. "1 2 4")
K_SHOTS="1"

# Build the --categories argument (omit it entirely when running all categories)
CAT_ARGS=()
if [ -n "$CATEGORIES" ]; then
    CAT_ARGS=(--categories $CATEGORIES)
fi

echo "--- Starting k-shot experiments for VisA ---"
for k in $K_SHOTS
do
    echo "--- Running VisA k=$k | categories: ${CATEGORIES:-all} ---"
    python -u main.py \
        --dataset_name visa \
        --dataset_path "$VISA_PATH" \
        "${CAT_ARGS[@]}" \
        --image_res 672 \
        --k_shot $k \
        --layers="-12,-13,-14,-15,-16,-17,-18" \
        --model_ckpt "facebook/dinov2-with-registers-giant" \
        --aug_count 30 \
        --pca_ev 0.99 \
        --agg_method "mean" \
        --seed 42 \
        --outdir "few_shot_results/results_k${k}_visa_dinov2G" \
        --kmeans_clusters 1 
done

echo "--- All experiments complete ---"


# 当kmeans_clusters = 1时，和原流程并无区别