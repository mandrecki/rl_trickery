python simple_train.py -m name=ff \
agent=a2c_image agent.network_params.architecture=crnn \
env=mazelab \
agent.network_params.random_cog_fraction=0.0,0.0,0.0,0.0,0.0 \
agent.algo_params.update_cognitive_values=false \
agent.network_params.append_a_cog=true
