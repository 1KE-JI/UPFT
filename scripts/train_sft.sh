set -x

# conda create -n torch240 python=3.10.13
# conda activate torch240
# conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# pip install openrlhf
# pip install wandb accelerate bitsandbytes deepspeed==0.15.0 isort jsonlines loralib optimum peft tensorboard torchmetrics transformers-stream-generator
# pip install antlr4-python3-runtime==4.11.0

planning_pruning=$1
prefix_length=$2
planning_pruning_mode=$3
DATASET_NAME=$4
train_model_type=$5
planning_pruning_ratio=$6

NUM_NODES=1
MAX_SEQ_LENGTH=3072
DATASET_NAME=Math

micro_train_batch_size=1
zero_stage=3
WORK_DIR=/apdcephfs_qy3/share_301812049/username/workspace/code/UPFT
MODEL_PATH=/apdcephfs_qy3/share_301812049/username/workspace/hf_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
judge_llm_path=/apdcephfs_qy3/share_301812049/username/workspace/hf_models/KbsdJames/Omni-Judge
OUTPUT_DIR=ckpts/DeepSeek-R1-Distill-Qwen-7B/${DATASET_NAME}_sft/UPFT

if [ "$train_model_type" = "llama" ]; then
    CHAT_TEMPLATE_NAME=llama-3.1-chat
elif [ "$train_model_type" = "qwen" ]; then
    CHAT_TEMPLATE_NAME=qwen-math
elif [ "$train_model_type" = "deepseek_R1" ]; then
    CHAT_TEMPLATE_NAME=deepseek_R1
else
    echo "model type error"
    exit 3
fi

save_only=0

data_ratio=-1

DATA_PATH=""
echo $DATA_PATH

if [[ "$DATASET_NAME" == *"1W"* ]]; then
    SAVE_STEPS=500
elif [[ "$DATASET_NAME" == *"limo"* ]]; then
    SAVE_STEPS=-1
else
    SAVE_STEPS=500
fi

ADD_SFT_PROMPT="1"
ADD_PLANNING_PROMPT=" Please provide the initial step towards resolving the question. This step may serve as a foundation but might not encompass the entire solution."

planning_pruning_token=1
DEBUG=0
max_epochs=1

if echo "$DATASET_NAME" | grep -q -- "prm12k"; then
    max_epochs=2
    MAX_SEQ_LENGTH=3072
    zero_stage=3
    micro_train_batch_size=1
fi

if echo "$DATASET_NAME" | grep -q -- "limo"; then
    max_epochs=3
    MAX_SEQ_LENGTH=4096
fi

if [ "$train_model_type" = "deepseek_R1" ]; then
    MAX_SEQ_LENGTH=16384
    # MAX_SEQ_LENGTH=12000
fi

if [ "$save_only" == "1" ]; then
    max_epochs=1
fi

LOAD_CHECKPOINT=1

if echo "$DATASET_NAME" | grep -q -- "negative"; then
    TASK=negative
    data_type="negative"
    negative_mode_list=("negative_whole")
else
    TASK=sft
    data_type="positive"
    negative_mode_list=("negative_none")
fi

if echo "$DATASET_NAME" | grep -q -- "posterior"; then
    TASK=noposterior
    data_type="negative"
    negative_mode_list=("noposterior")
fi

for negative_mode in "${negative_mode_list[@]}"; do
    planning_prefix_tuning_length=${prefix_length}
    planning_suffix_tuning_length=0
    lr=2e-6
    lr_s=constant_with_warmup

    planning_suffix=planning_pruning_${planning_pruning}_ratio_${planning_pruning_ratio}_planning_pruning_token_${planning_pruning_token}
    SUFFIX=${data_type}_epoch_${max_epochs}_lr${lr}_ratio_${data_ratio}_${DATASET_NAME}_${CHAT_TEMPLATE_NAME}_pl_${planning_prefix_tuning_length}

    if [ "$planning_pruning" = "1" ]; then
        SUFFIX=${SUFFIX}_${planning_suffix}_mixture_ppm_${planning_pruning_mode}
    fi

    if [ "$data_type" = "negative" ]; then
        SUFFIX=${SUFFIX}_${negative_mode}
    fi
    if [ "$DEBUG" = "1" ]; then
        SUFFIX=debug_${SUFFIX}
        rm -rf "${OUTPUT_DIR}/${SUFFIX}"
    fi

    if [ "$train_model_type" = "llama" ]; then
        RUN_NAME=Llama-3.1-8B-Instruct_${SUFFIX}
    elif [ "$train_model_type" = "qwen" ]; then
        RUN_NAME=Qwen2.5-Math-7B-Instruct_${SUFFIX}
    elif [ "$train_model_type" = "deepseek_R1" ]; then
        RUN_NAME=DeepSeek-R1-Distill-Qwen-7B_${SUFFIX}
    fi

    if [ -d "${OUTPUT_DIR}/${SUFFIX}" ]; then
        cp ${OUTPUT_DIR}/${SUFFIX}/training.log ${OUTPUT_DIR}/${SUFFIX}/training.log.bak
    fi
    if [ -e "${OUTPUT_DIR}/${SUFFIX}/model-00001-of-00004.safetensors" ]; then
        LOAD_CHECKPOINT=0
        if [ "$save_only" = "1" ]; then
            LOAD_CHECKPOINT=1
        fi
    fi

    if [[ -d "${OUTPUT_DIR}/${SUFFIX}" && "$LOAD_CHECKPOINT" = "0" ]]; then
        echo "path exist: ${OUTPUT_DIR}/${SUFFIX}"
    else
        echo "path: ${OUTPUT_DIR}/${SUFFIX}"
        mkdir -p ${OUTPUT_DIR}/${SUFFIX}

        echo "train on 1 node with 8 GPUs"

        for i in {0..10}; do
            deepspeed --module train.train_sft \
                --max_len ${MAX_SEQ_LENGTH} \
                --dataset ${DATASET_NAME} \
                --input_key question \
                --output_key response \
                --debug ${DEBUG} \
                --train_batch_size 64 \
                --micro_train_batch_size ${micro_train_batch_size} \
                --lr_scheduler ${lr_s} \
                --max_samples 500000 \
                --pretrain ${MODEL_PATH} \
                --save_path ${OUTPUT_DIR}/${SUFFIX} \
                --save_steps ${SAVE_STEPS} \
                --logging_steps 1 \
                --adam_offload \
                --eval_steps -1 \
                --zero_stage ${zero_stage} \
                --bf16 \
                --flash_attn \
                --max_epochs ${max_epochs} \
                --learning_rate ${lr} \
                --load_checkpoint \
                --gradient_checkpointing \
                --chat_template_name ${CHAT_TEMPLATE_NAME} \
                --apply_chat_template \
                --tasks $TASK \
                --data_ratio ${data_ratio} \
                --data_path ${DATA_PATH} \
                --add_prompt "${ADD_SFT_PROMPT}" \
                --add_planning_prompt "${ADD_PLANNING_PROMPT}" \
                --planning_pruning_ratio "${planning_pruning_ratio}" \
                --negative_mode ${negative_mode} \
                --dataset_name ${DATASET_NAME} \
                --save_only ${save_only} \
                --planning_pruning ${planning_pruning} \
                --planning_pruning_token ${planning_pruning_token} \
                --planning_pruning_mode ${planning_pruning_mode} \
                --planning_prefix_tuning_length ${planning_prefix_tuning_length} 2>&1 | tee ${OUTPUT_DIR}/${SUFFIX}/training.log
            if [ -e "${OUTPUT_DIR}/${SUFFIX}/model-00001-of-00004.safetensors" ]; then
                echo "train success, exit"
                break
            fi
        done
    fi

    ADD_PROMPT=${ADD_SFT_PROMPT}
    GEN_DIR=${WORK_DIR}/gen

    MODEL_NAME=${RUN_NAME}
    EXP_DIR=${GEN_DIR}/$MODEL_NAME

    EVAL_DATASETS=(gsm8k math500 aime24 gpqa)

    for split in "${EVAL_DATASETS[@]}"; do
        split=${split}
        MATH_EXP_DIR=${EXP_DIR}/${split}
        EVAL_MODEL=${OUTPUT_DIR}/${SUFFIX}
        if [ -e "${MATH_EXP_DIR}/config.json" ]; then
            echo "file exists: ${MATH_EXP_DIR}/config.json"
        else
            echo "file does not exist: $MATH_EXP_DIR/config.json, begin evaluation..."

            python3 ${WORK_DIR}/inference/eval.py --type eval \
                --base_model ${EVAL_MODEL} \
                --chat_template_name ${CHAT_TEMPLATE_NAME} \
                --output_dir $MATH_EXP_DIR \
                --bf16 False \
                --split ${split} \
                --llm_judge True \
                --add_prompt "${ADD_PROMPT}"
        fi

        cat ${MATH_EXP_DIR}/config.json
        if [ -e "${MATH_EXP_DIR}/result.log" ]; then
            echo "file exists: ${MATH_EXP_DIR}/result.log"
        else
            python3 $WORK_DIR/inference/eval.py --type judge \
                --config_file ${MATH_EXP_DIR}/config.json \
                --judge_llm_path ${judge_llm_path}
        fi
        echo "----------------------- ${split} done -----------------------"
        cat $MATH_EXP_DIR/result.log
    done
done
