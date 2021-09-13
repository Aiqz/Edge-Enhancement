DATA160=/hdd/aiqingzhong/Imagenet-sz/160
DATA352=/hdd/aiqingzhong/Imagenet-sz/352
DATA=/hdd/aiqingzhong/Imagenet

NAME=eps2_ee

CONFIG1=configs_ee/configs_fast_2px_phase1_ee.yml
CONFIG2=configs_ee/configs_fast_2px_phase2_ee.yml
CONFIG3=configs_ee/configs_fast_2px_phase3_ee.yml

PREFIX1=fast_adv_phase1_${NAME}
PREFIX2=fast_adv_phase2_${NAME}
PREFIX3=fast_adv_phase3_${NAME}

OUT1=fast_train_phase1_${NAME}.out
OUT2=fast_train_phase2_${NAME}.out
OUT3=fast_train_phase3_${NAME}.out

EVAL1=fast_eval_phase1_${NAME}.out
EVAL2=fast_eval_phase2_${NAME}.out
EVAL3=fast_eval_phase3_${NAME}.out

END1=/hdd/aiqingzhong/code21/Freq_Contour/FGSM_ImageNet/trained_models/fast_adv_phase1_eps2_ee_step2_eps2_repeat1/checkpoint_epoch6.pth.tar
END2=/hdd/aiqingzhong/code21/Freq_Contour/FGSM_ImageNet/trained_models/fast_adv_phase2_eps2_ee_step2_eps2_repeat1/checkpoint_epoch12.pth.tar
END3=/hdd/aiqingzhong/code21/Freq_Contour/FGSM_ImageNet/trained_models/fast_adv_phase3_eps2_ee_step2_eps2_repeat1/checkpoint_epoch15.pth.tar

# training for phase 1
python -u main_fast.py $DATA160 -c $CONFIG1 --output_prefix $PREFIX1 | tee $OUT1

# evaluation for phase 1
# python -u main_fast.py $DATA160 -c $CONFIG1 --output_prefix $PREFIX1 --resume $END1  --evaluate | tee $EVAL1

# training for phase 2
python -u main_fast.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --resume $END1 | tee $OUT2

# evaluation for phase 2
# python -u main_fast.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --resume $END2 --evaluate | tee $EVAL2

# training for phase 3
python -u main_fast.py $DATA -c $CONFIG3 --output_prefix $PREFIX3 --resume $END2 | tee $OUT3

# evaluation for phase 3
# python -u main_fast.py $DATA -c $CONFIG3 --output_prefix $PREFIX3 --resume $END3 --evaluate | tee $EVAL3
python main_fast.py /hdd/aiqingzhong/Imagenet --config configs_ee/configs_fast_2px_evaluate_ee.yml --output_prefix eval_2px_ee --resume trained_models/fast_adv_phase3_eps2_ee_step2_eps2_repeat1/model_best.pth.tar --evaluate --restarts 10