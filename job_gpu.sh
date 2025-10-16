#!/bin/bash
#SBATCH --job-name=generation
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --export=ALL
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --output=/pasteur/appa/homes/bsow/cued_speech/generation_%A_%a.log
#SBATCH --time=01:00:00

# Run the Python script with the current parameters
# run the cued-speech generate download/test_generate.mp4 command
cued-speech generate download/test_generate.mp4
