nohup python train.py --yaml_file args_yml/model_base_IRL/hopper_v2_medexp.yml --seed 3 --uuid hopper_test1 >>1.out &
nohup python train.py --yaml_file args_yml/model_base_IRL/halfcheetah_v2_medexp.yml --seed 3 --uuid halfcheetah_test1 >>2.out &
nohup python train.py --yaml_file args_yml/model_base_IRL/walker2d_v2_medexp.yml --seed 3 --uuid walker_test1 >>3.out &