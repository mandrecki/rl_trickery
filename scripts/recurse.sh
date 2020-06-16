python simple_train.py -m \
agent=a2c_image agent.network_params.architecture=ff \
env=mazelab \
agent.network_params.random_cog_fraction=0.0,0.05,0.2,0.5,0.8
agent.network_params.update_cognitive_values=true,false

python simple_train.py -m \
agent=a2c_image agent.network_params.architecture=rnn \
env=mazelab \
agent.network_params.random_cog_fraction=0.0,0.05,0.2,0.5,0.8
agent.network_params.update_cognitive_values=true,false

python simple_train.py -m \
agent=a2c_image agent.network_params.architecture=crnn \
env=mazelab \
agent.network_params.random_cog_fraction=0.0,0.05,0.2,0.5,0.8 \
agent.network_params.update_cognitive_values=true,false
