#python simple_train.py -m name=crnn_random_recu \
#agent=a2c_image agent.network_params.architecture=crnn \
#env=mazelab \
#agent.network_params.random_cog_fraction=0.0,0.5,0.8 \
#agent.algo_params.update_cognitive_values=false \
#agent.network_params.append_a_cog=true

python simple_train.py -m name=crnn_fixed_recu \
agent=a2c_image agent.network_params.architecture=crnn \
env=mazelab \
agent.network_params.fixed_recursive_depth=10,2,5,1 \
agent.algo_params.update_cognitive_values=false \
agent.network_params.append_a_cog=true

python simple_train.py -m name=rnn_random_recu \
agent=a2c_image agent.network_params.architecture=rnn \
env=mazelab \
agent.network_params.random_cog_fraction=0.0,0.5,0.8 \
agent.algo_params.update_cognitive_values=false \
agent.network_params.append_a_cog=true

python simple_train.py -m name=rnn_fixed_recu \
agent=a2c_image agent.network_params.architecture=rnn \
env=mazelab \
agent.network_params.fixed_recursive_depth=1,2,5,10 \
agent.algo_params.update_cognitive_values=false \
agent.network_params.append_a_cog=true

