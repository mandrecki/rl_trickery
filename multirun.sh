python simple_train.py -m agent=a2c_image env=mazelab \
seed=int:1:1000000 \
agent.num_steps=log:int:5:30 \
agent.algo_params.use_timeout=true,false \
agent.algo_params.optimizer_type=adam,rmsprop \
agent.algo_params.lr=log:0.0001:0.01 \
agent.algo_params.max_grad_norm=log:0.01:10.0 \
agent.num_envs=log:int:1:32 \
agent.network_params.state_channels=log:int:8:128
#agent.algo_params.gamma=log:0.95:0.999 \

