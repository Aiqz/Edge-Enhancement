# 4eps evaluation
python main_fast.py /home/ubuntu/datasets/seqres --config configs/configs_fast_4px_evaluate.yml --output_prefix eval_4px --resume trained_models/fast_adv_phase3_eps4_step5_eps4_repeat1/model_best.pth.tar --evaluate --restarts 1

# 2eps evaluation 
# python main_fast.py /hdd/aiqingzhong/Imagenet --config configs/configs_fast_2px_evaluate.yml --output_prefix eval_2px --resume trained_models/fast_adv_phase3_eps2_step2_eps2_repeat1/model_best.pth.tar --evaluate --restarts 10
