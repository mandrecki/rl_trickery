python train.py -m name=baseline_rnn \
agent=a2c_proprio agent.network_params.architecture=rnn \
env=cartpole \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false

