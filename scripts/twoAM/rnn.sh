python train.py -m name=cp_rnn_twoAM \
agent=a2c_proprio agent.network_params.architecture=rnn \
env=cartpole \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=true

python train.py -m name=cp_crnn_twoAM \
agent=a2c_proprio agent.network_params.architecture=crnn \
env=cartpole \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=true

