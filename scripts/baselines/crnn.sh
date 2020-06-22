#python train.py -m name=m11fixed_crnn \
#agent=a2c_image agent.network_params.architecture=crnn \
#env=mazelab \
#env.env_kwargs.maze_size=9 \
#env.env_kwargs.maze_fixed=false \
#env.env_kwargs.goal_fixed=false \
#seed=0,0,0,0,0 \
#agent.algo_params.twoAM=false \
#
#python train.py -m name=m11fixed_rnn \
#agent=a2c_image agent.network_params.architecture=rnn \
#env=mazelab \
#env.env_kwargs.maze_size=9 \
#env.env_kwargs.maze_fixed=false \
#env.env_kwargs.goal_fixed=false \
#seed=0,0,0,0,0 \
#agent.algo_params.twoAM=false \


python train.py -m name=m9_rnn_2am \
agent=a2c_image agent.network_params.architecture=rnn \
env=mazelab \
env.env_kwargs.maze_size=9 \
env.env_kwargs.maze_fixed=false \
env.env_kwargs.goal_fixed=false \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=true \


python train.py -m name=m9_crnn_2am \
agent=a2c_image agent.network_params.architecture=crnn \
env=mazelab \
env.env_kwargs.maze_size=9 \
env.env_kwargs.maze_fixed=false \
env.env_kwargs.goal_fixed=false \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=true \

