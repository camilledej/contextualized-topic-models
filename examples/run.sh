## use nvdia-smi to determine the GPUs available
## modify this to be the GPU number you want to use
export CUDA_VISIBLE_DEVICES="0"

python zeroshot_multilingual.py
