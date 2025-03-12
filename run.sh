stage=$1
model=$2
dataset=$3
suffix=$4
max_length=$5
shift; shift; shift; shift; shift
gpu_ids=("$@")
icl_use=true
vllm_use=true
echo "Inferencing on model: $model, dataset: $dataset, suffix: $suffix, gpu_ids: ${gpu_ids[@]}"
split_num=${#gpu_ids[@]}


for ((i=0; i<$split_num; i++)); do
    if [ $stage = sample ]; then
        echo "Inferencing on GPU ${gpu_ids[i]} ..."
        CUDA_VISIBLE_DEVICES=${gpu_ids[i]} python code/sample.py --model_name $model \
                                                                --dataset $dataset \
                                                                --max_length $max_length \
                                                                --icl_use $icl_use \
                                                                --split_num $split_num \
                                                                --split_id $i \
                                                                --model_suffix $suffix &

    elif [ $stage = infer ]; then
        echo "Inferencing on GPU ${gpu_ids[i]} ..."
        CUDA_VISIBLE_DEVICES=${gpu_ids[i]} python code/infer.py --model_name $model \
                                                                --dataset $dataset \
                                                                --max_length $max_length \
                                                                --icl_use $icl_use \
                                                                --vllm_use $vllm_use \
                                                                --split_num $split_num \
                                                                --split_id $i \
                                                                --model_suffix $suffix &

    fi

done

wait



if [ $stage = sample ]; then
    echo "Sampling done!"
    exit
elif [ $stage = infer ]; then
    echo "Inference done!"
    python code/eval.py --model_name $model --dataset $dataset --model_suffix $suffix --icl_use $icl_use --vllm_use $vllm_use --split_num $split_num
fi

echo "Inference done!"
