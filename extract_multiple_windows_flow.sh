export LD_LIBRARY_PATH=''
export CUDA_VISIBLE_DEVICES='1'
seq 3 2 7 | parallel -u -j12 'python extract_features.py -mode flow -load_model /media/storage2/dominic/i3d-asegmentation/models/flow_imagenet.pt -save_dir /media/storage2/dominic/ms-tcn/data/BL_and_PL/new_features/temporal_window_{}/flow -temporal_window {}'


