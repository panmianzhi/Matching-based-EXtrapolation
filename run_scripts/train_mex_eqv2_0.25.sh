#!/bin/bash
export OMP_NUM_THREADS=1

cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
if [ -z "$cuda_visible_devices" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set."
    NUM_GPUS=0
else
    IFS=',' read -r -a gpu_array <<< "$cuda_visible_devices"
    NUM_GPUS=${#gpu_array[@]}
    echo "NUM_GPUS="$NUM_GPUS
fi

################### (batch_size lr wd) ######################
castelli_eform_small_param=(64 0.0001 0)
mp_gvrh_small_param=(64 0.0001 0) # best b=64
mp_gvrh_large_param=(64 0.0001 0) # best b=64
mp_dielectric_small_param=(64 0.0001 0) #best b=64
mp_dielectric_large_param=(64 0.0001 0)
mp_phonons_small_param=(64 0.001 0.001)
mp_phonons_large_param=(32 0.001 0.001)

####################### all tasks ########################
cfg_names=(
    "castelli_eform_small"
    "mp_gvrh_small"
    "mp_gvrh_large"
    "mp_dielectric_small"
    "mp_dielectric_large"
    "mp_phonons_small"
    "mp_phonons_large"
)

cfg_root_dir="configs/mex-0.25" # check the config to ensure the same algorithm as cur_algo
cur_algo="mex-0.25_cos-mlp_nce-gradclip=10"
m_port=2062
d_port=5093

for cfg_n in "${cfg_names[@]}"; do
    declare -n param_array="${cfg_n}_param" 

    bs=${param_array[0]}
    (( bs = bs / $NUM_GPUS )) # DDP
    lr=${param_array[1]}
    wd=${param_array[2]}
    echo "cfg: "$cfg_n "bs: "$bs "lr: "$lr "wd: "$wd

    python build_param_configs.py --bs $bs --lr $lr --wd $wd --tgt-dir $cfg_root_dir
    for seed in {0..2}; do
        torchrun \
        --nproc_per_node=$NUM_GPUS --master_port=$m_port ocp/main.py \
        --mode train \
        --early-stop 20 \
        --config-yml ${cfg_root_dir}/${cfg_n}.yml \
        --identifier seed=${seed} \
        --seed $seed \
        --run-dir record_extrapolation_v2/${cur_algo}/eqv2-scr_lr=${lr}_wd=${wd}_bs=${NUM_GPUS}-${bs}/${cfg_n} \
        --amp \
        --num-gpus $NUM_GPUS \
        --distributed \
        --distributed-port $d_port
    done
done