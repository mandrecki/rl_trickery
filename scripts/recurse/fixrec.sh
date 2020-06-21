python train.py -m name=cp_crnn_fixrec_5 \
agent=a2c_image agent.network_params.architecture=crnn \
env=mazelab \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false \
agent.network_params.fixed_recursive_depth=5

python train.py -m name=cp_rnn_fixrec_5 \
agent=a2c_image agent.network_params.architecture=rnn \
env=mazelab \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false \
agent.network_params.fixed_recursive_depth=5

python train.py -m name=cp_crnn_fixrec_2 \
agent=a2c_image agent.network_params.architecture=crnn \
env=mazelab \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false \
agent.network_params.fixed_recursive_depth=2

python train.py -m name=cp_rnn_fixrec_2 \
agent=a2c_image agent.network_params.architecture=rnn \
env=mazelab \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false \
agent.network_params.fixed_recursive_depth=2

