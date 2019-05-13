export LD_LIBRARY_PATH=''
export CUDA_VISIBLE_DEVICES='0'
seq 3 2 7 | parallel -u -j12 'python extract_features.py -save_dir /media/storage2/dominic/ms-tcn/data/BL_and_PL/new_features/temporal_window_{}/RGB -temporal_window {};'

