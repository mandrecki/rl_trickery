python simple_train.py -m name=cp_crnn_fixed_recurse \
num_timesteps=1e5 \
agent=a2c_proprio \
env=cartpole \
agent.network_params.fixed_recursive_depth=1,1,1,2,2,2,5,5,5 \
agent.network_params.append_a_cog=true

python simple_train.py -m name=cp_crnn_rand_recurse \
num_timesteps=1e5 \
agent=a2c_proprio \
env=cartpole \
agent.network_params.random_cog_fraction=0,0,0,0.5,0.5,0.5,0.8,0.8,0.8 \
agent.network_params.append_a_cog=true
