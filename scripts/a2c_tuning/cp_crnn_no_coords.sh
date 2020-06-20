python train.py -m name=cp_crnn_no_coords \
agent=a2c_proprio agent.network_params.architecture=crnn \
env=cartpole \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false \
agent.network_params.append_coords=false

