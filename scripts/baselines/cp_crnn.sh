python train.py -m name=baseline_crnn \
agent=a2c_proprio agent.network_params.architecture=crnn \
env=cartpole \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false

