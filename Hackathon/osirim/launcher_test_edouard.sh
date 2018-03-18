#!/bin/sh


#SBATCH --job-name=keras_model_training
#SBATCH --output=ML-%j-Theano.out
#SBATCH --error=ML-%j-Theano.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodelist=gpu-nc07

srun keras-py3-th python "$HOME/Hackathon_model_1_edouard.py"

