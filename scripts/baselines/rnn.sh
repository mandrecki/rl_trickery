python train.py -m name=m13_baseline_rnn \
agent=a2c_image agent.network_params.architecture=rnn \
env=mazelab \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false

python train.py -m name=m13_baseline_crnn \
agent=a2c_image agent.network_params.architecture=crnn \
env=mazelab \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false

