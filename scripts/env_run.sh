ENV=m15
AGENT=a2c_image

# base
python train.py -m name=ff \
agent=${AGENT} agent.network_params.architecture=ff \
env=${ENV} \
seed=0,0,0 \

python train.py -m name=crnn \
agent=${AGENT} agent.network_params.architecture=crnn \
env=${ENV} \
seed=0,0,0 \

python train.py -m name=rnn \
agent=${AGENT} agent.network_params.architecture=rnn \
env=${ENV} \
seed=0,0,0 \

# 3 recurse
python train.py -m name=3crnn \
agent=${AGENT} agent.network_params.architecture=crnn \
env=${ENV} \
seed=0,0,0 \
agent.network_params.fixed_recursive_depth=3

python train.py -m name=3rnn \
agent=${AGENT} agent.network_params.architecture=rnn \
env=${ENV} \
seed=0,0,0 \
agent.network_params.fixed_recursive_depth=3

# 9 recurse
python train.py -m name=9crnn \
agent=${AGENT} agent.network_params.architecture=crnn \
env=${ENV} \
seed=0,0,0 \
agent.network_params.fixed_recursive_depth=9

python train.py -m name=9rnn \
agent=${AGENT} agent.network_params.architecture=rnn \
env=${ENV} \
seed=0,0,0 \
agent.network_params.fixed_recursive_depth=9
