ENV=m15fixed
AGENT=a2c_image

# base
#python train.py -m name=ff \
#agent=${AGENT} agent.network_params.architecture=ff \
#env=${ENV} \
#seed=0,0,0 \

#python train.py -m name=crnn \
#agent=${AGENT} agent.network_params.architecture=crnn \
#env=${ENV} \
#seed=0,0,0 \

#python train.py -m name=3crnn \
#agent=${AGENT} agent.network_params.architecture=crnn \
#env=${ENV} \
#agent.network_params.fixed_recursive_depth=3 \
#seed=0,0,0 \

#python train.py -m name=crnn_amnesia \
#agent=${AGENT} agent.network_params.architecture=crnn \
#env=${ENV} \
#seed=0,0,0 \
#agent.network_params.amnesia=true \

#python train.py -m name=3crnn_amnesia \
#agent=${AGENT} agent.network_params.architecture=crnn \
#env=${ENV} \
#agent.network_params.amnesia=true \
#agent.network_params.fixed_recursive_depth=3 \
#seed=0,0,0 \

#python train.py -m name=9crnn \
#agent=${AGENT} agent.network_params.architecture=crnn \
#env=${ENV} \
#seed=0,0,0 \
#agent.network_params.fixed_recursive_depth=9

#python train.py -m name=rnn \
#agent=${AGENT} agent.network_params.architecture=rnn \
#env=${ENV} \
#seed=0,0,0 \

#python train.py -m name=3rnn \
#agent=${AGENT} agent.network_params.architecture=rnn \
#env=${ENV} \
#seed=0,0,0 \
#agent.network_params.fixed_recursive_depth=3

#python train.py -m name=rnn_amnesia \
#agent=${AGENT} agent.network_params.architecture=rnn \
#agent.network_params.amnesia=true \
#env=${ENV} \
#seed=0,0,0 \

#python train.py -m name=3rnn_amnesia \
#agent=${AGENT} agent.network_params.architecture=rnn \
#env=${ENV} \
#agent.network_params.amnesia=true \
#seed=0,0,0 \
#agent.network_params.fixed_recursive_depth=3


#python train.py -m name=9rnn \
#agent=${AGENT} agent.network_params.architecture=rnn \
#env=${ENV} \
#seed=0,0,0 \
#agent.network_params.fixed_recursive_depth=9

python train.py -m name=crnn_2am \
agent=${AGENT} \
env=${ENV} \
agent.network_params.architecture=crnn \
agent.network_params.detach_cognition=true \
agent.network_params.two_transitions=true \
agent.network_params.append_a_cog=false \
agent.algo_params.twoAM=true \
agent.algo_params.gamma_cog=0.95 \
agent.algo_params.cognitive_coef=1 \
agent.algo_params.cognitive_cost=0.005 \
agent.algo_params.cognitive_rewards=D \
agent.algo_params.update_cognitive_values=false \
seed=0 \
