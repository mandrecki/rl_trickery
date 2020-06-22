# base
python train.py -m name=m11fixed_crnn_base \
agent=a2c_image agent.network_params.architecture=crnn \
env=m11fixed \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false \

python train.py -m name=m11fixed_rnn_base \
agent=a2c_image agent.network_params.architecture=rnn \
env=m11fixed \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false \


# 2 recurse
python train.py -m name=m11fixed_crnn_fixrec_2 \
agent=a2c_image agent.network_params.architecture=crnn \
env=m11fixed \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false \
agent.network_params.fixed_recursive_depth=2

python train.py -m name=m11fixed_rnn_fixrec_2 \
agent=a2c_image agent.network_params.architecture=rnn \
env=m11fixed \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false \
agent.network_params.fixed_recursive_depth=2

# 5 recurse
python train.py -m name=m11fixed_crnn_fixrec_5 \
agent=a2c_image agent.network_params.architecture=crnn \
env=m11fixed \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false \
agent.network_params.fixed_recursive_depth=5

python train.py -m name=m11fixed_rnn_fixrec_5 \
agent=a2c_image agent.network_params.architecture=rnn \
env=m11fixed \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false \
agent.network_params.fixed_recursive_depth=5

# 2 random recurse
python train.py -m name=m11fixed_crnn_randrec_2 \
agent=a2c_image agent.network_params.architecture=crnn \
env=m11fixed \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false \
agent.network_params.random_cog_fraction=0.5 \

python train.py -m name=m11fixed_rnn_randrec_2 \
agent=a2c_image agent.network_params.architecture=rnn \
env=m11fixed \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false \
agent.network_params.random_cog_fraction=0.5 \
