#!/bin/bash
#----------------------------------
# Specifying grid engine options
#----------------------------------
#$ -S /bin/bash  
# the working directory where the commands below will
# be executed: (make sure to specify)
#$ -wd /data/users/abhavan/hatespeech/
#
# logging files will go here: (make sure to specify)
#$ -e /data/users/abhavan/logs/ -o /data/users/abhavan/logs/
#  
# Resource specifications
# Specify maximum memory usage: (optional)
#$ -l s_vmem=500G 
#$ -l hostname=cl14lx
#----------------------------------
#  Running some bash commands 
#----------------------------------
export PATH="/nethome/abhavan/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda"
source activate anjenv

pwd  
echo "hello" 
#----------------------------------
# Running your code (here we run some python script as an example)
#----------------------------------
#echo "Embeddings based on mean of sequences"
python train_svm.py



