# base
python train.py -m name=m15_rnn_base \
agent=a2c_image agent.network_params.architecture=rnn \
env=m15 \
seed=0,0 \
agent.algo_params.twoAM=false \

python train.py -m name=m15_crnn_base \
agent=a2c_image agent.network_params.architecture=crnn \
env=m15 \
seed=0,0 \
agent.algo_params.twoAM=false \

# 2 recurse
python train.py -m name=m15_rnn_fixrec_2 \
agent=a2c_image agent.network_params.architecture=rnn \
env=m15 \
seed=0,0 \
agent.algo_params.twoAM=false \
agent.network_params.fixed_recursive_depth=2

python train.py -m name=m15_crnn_fixrec_2 \
agent=a2c_image agent.network_params.architecture=crnn \
env=m15 \
seed=0,0 \
agent.algo_params.twoAM=false \
agent.network_params.fixed_recursive_depth=2

# 2 random recurse
python train.py -m name=m15_rnn_randrec_2 \
agent=a2c_image agent.network_params.architecture=rnn \
env=m15 \
seed=0,0 \
agent.algo_params.twoAM=false \
agent.network_params.random_cog_fraction=0.5 \


python train.py -m name=m15_crnn_randrec_2 \
agent=a2c_image agent.network_params.architecture=crnn \
env=m15 \
seed=0,0 \
agent.algo_params.twoAM=false \
agent.network_params.random_cog_fraction=0.5 \
