python simple_train.py -m name=cp_2am_crnn \
env=cartpole \
num_timesteps=1e5 \
agent=a2c_proprio \
agent.algo_params.twoAM=true \
agent.algo_params.cognition_cost=0.
