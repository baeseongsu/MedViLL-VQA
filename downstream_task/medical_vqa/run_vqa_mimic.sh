PATH_ITR=/home/dekyung/seongsu_temp/MedViLL-VQA/downstream_task/medical_vqa/pretrained_model/par/pytorch_model.bin
EXP_NAME=medvill_test_chest

CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port 9873 \
--use_env ./downstream_task/medical_vqa/finetune_vqa_mimic.py \
--model_recover_path ${PATH_ITR} \
--vqa_rad all \
--num_train_epochs 50 \
--train_batch_size 16 \
--wandb \
--exp_name=${EXP_NAME}