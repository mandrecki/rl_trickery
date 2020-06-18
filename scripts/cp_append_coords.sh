python simple_train.py -m name=crnn_append_coords \
num_timesteps=1e5 \
agent=a2c_proprio \
env=cartpole \
agent.network_params.append_coords=true,true,true,true,true,false,false,false,false,false
