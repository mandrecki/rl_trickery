# base
python train.py -m name=pc_ff_base \
agent=a2c_image agent.network_params.architecture=ff \
env=pc \
seed=0,0,0 \

python train.py -m name=pc_rnn_base \
agent=a2c_image agent.network_params.architecture=rnn \
env=pc \
seed=0,0,0 \

python train.py -m name=pc_crnn_base \
agent=a2c_image agent.network_params.architecture=crnn \
env=pc \
seed=0,0,0 \

# 2 fix recurse
python train.py -m name=pc_rnn_fixrec2 \
agent=a2c_image agent.network_params.architecture=rnn \
env=pc \
seed=0,0,0 \
agent.network_params.fixed_recursive_depth=2 \

python train.py -m name=pc_crnn_fixrec2 \
agent=a2c_image agent.network_params.architecture=crnn \
env=pc \
seed=0,0,0 \
agent.network_params.fixed_recursive_depth=2 \

# 2 random recurse
python train.py -m name=pc_rnn_randrec2_2trans \
agent=a2c_image agent.network_params.architecture=rnn \
env=pc \
seed=0,0,0 \
agent.network_params.random_cog_fraction=0.5 \
agent.network_params.two_transitions=true \
agent.network_params.append_a_cog=false

python train.py -m name=pc_crnn_randrec2_2trans \
agent=a2c_image agent.network_params.architecture=crnn \
env=pc \
seed=0,0,0 \
agent.network_params.random_cog_fraction=0.5 \
agent.network_params.two_transitions=true \
agent.network_params.append_a_cog=false

