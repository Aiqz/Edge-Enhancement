# 4eps evaluation
python main_fast.py /home/ubuntu/datasets/seqres --config configs_hfs_canny/configs_fast_4px_evaluate_ee.yml --output_prefix eval_4px_ee --resume trained_models/fast_adv_phase3_eps4_ee_step5_eps4_repeat1/model_best.pth.tar --evaluate --restarts 10

# 2eps evaluation 
# python main_fast.py /home/ubuntu/datasets/seqres --config configs_hfs_canny/configs_fast_2px_evaluate_ee.yml --output_prefix eval_2px_ee --resume trained_models/fast_adv_phase3_eps2_ee_step2_eps2_repeat1/model_best.pth.tar --evaluate --restarts 10
