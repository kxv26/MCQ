#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --job-name=simple_job

# ensure that there are now left-over modules loaded from previous jobs
module purge

# load your previously created virtual environment
source ./venv/bin/activate

# make sure to load all needed modules after activating your virtual environment now
# this ensures that all the dependencies are loaded correctly inside the environment
module load PyTorch/2.1.2-foss-2022b

# move cached datasets to the /scratch directory
export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"

# move downloaded models and tokenizers to the /scratch directory
export TRANSFORMERS_CACHE="/scratch/$USER/.cache/huggingface/hub"

if [ ! -d "bleurt" ]; then
    git clone https://github.com/google-research/bleurt.git
fi
if [ -d "bleurt" ]; then
    cd bleurt
    pip3 install .
    cd ..
fi

pip3 install transformers datasets torch sentencepiece nltk rouge_score accelerate tf-keras evaluate bert_score

# start your program
python3 main.py evaluate
