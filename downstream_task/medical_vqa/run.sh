# PATH_ITR=/home/dekyung/seongsu_temp/MedViLL-VQA/downstream_task/medical_vqa/pretrained_model/par/pytorch_model.bin
# EXP_NAME=medvill_original_code_full
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
# --nproc_per_node=1 \
# --master_port 9872 \
# --use_env ./downstream_task/medical_vqa/finetune.py \
# --model_recover_path ${PATH_ITR} \
# --tasks vqa \
# --s2s_prob 0 \
# --bi_prob 1 \
# --mask_prob 0 \
# --vqa_rad all \
# --train_batch_size 16 \
# --exp_name=${EXP_NAME}

# PATH_ITR=/home/dekyung/seongsu_temp/MedViLL-VQA/downstream_task/medical_vqa/pretrained_model/par/pytorch_model.bin
# EXP_NAME=medvill_original_code_chest
# CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch \
# --nproc_per_node=1 \
# --master_port 9872 \
# --use_env ./downstream_task/medical_vqa/finetune.py \
# --model_recover_path ${PATH_ITR} \
# --tasks vqa \
# --s2s_prob 0 \
# --bi_prob 1 \
# --mask_prob 0 \
# --vqa_rad chest \
# --num_train_epochs 50 \
# --train_batch_size 16 \
# --exp_name=${EXP_NAME}

# PATH_ITR=/home/dekyung/seongsu_temp/MedViLL-VQA/downstream_task/medical_vqa/pretrained_model/par/pytorch_model.bin
# EXP_NAME=medvill_original_code_all
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
# --nproc_per_node=1 \
# --master_port 9873 \
# --use_env ./downstream_task/medical_vqa/finetune.py \
# --model_recover_path ${PATH_ITR} \
# --tasks vqa \
# --s2s_prob 0 \
# --bi_prob 1 \
# --mask_prob 0 \
# --vqa_rad all \
# --num_train_epochs 50 \
# --train_batch_size 16 \
# --exp_name=${EXP_NAME}


# PATH_ITR=/home/edlab/ssbae/mimic_mv_real/mimic-cxr/downstream_model/revision/vqa/pretrained_model_mimic-cxr_medvill_original_code_chest/model.50.bin
# EXP_NAME=medvill_test_chest

# CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch \
# --nproc_per_node=1 \
# --master_port 9872 \
# --use_env ./downstream_task/medical_vqa/finetune_modified.py \
# --model_recover_path ${PATH_ITR} \
# --tasks vqa \
# --s2s_prob 0 \
# --bi_prob 1 \
# --mask_prob 0 \
# --vqa_rad chest \
# --num_train_epochs 50 \
# --train_batch_size 16 \
# --exp_name=${EXP_NAME} \
# --vqa_eval

PATH_ITR=/home/dekyung/seongsu_temp/MedViLL-VQA/downstream_task/medical_vqa/pretrained_model/par/pytorch_model.bin
EXP_NAME=medvill_test_chest
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port 9872 \
--use_env ./downstream_task/medical_vqa/finetune_modified.py \
--model_recover_path ${PATH_ITR} \
--tasks vqa \
--s2s_prob 0 \
--bi_prob 1 \
--mask_prob 0 \
--vqa_rad chest \
--num_train_epochs 50 \
--train_batch_size 16 \
--exp_name=${EXP_NAME} \
--wandb \
--wandb_project phase1-qa-dataset-debugging-seongsu