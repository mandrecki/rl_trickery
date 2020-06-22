python train.py -m name=m9_crnn_twoAM \
agent=a2c_image agent.network_params.architecture=crnn \
env=mazelab \
env.env_kwargs.maze_size=9 \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=true \
agent.network_params.detach_cognition=true

python train.py -m name=m9_rnn_twoAM \
agent=a2c_image agent.network_params.architecture=rnn \
env=mazelab \
env.env_kwargs.maze_size=9 \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=true \
agent.network_params.detach_cognition=true
