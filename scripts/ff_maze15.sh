python simple_train.py -m name=crnn_2am \
agent=a2c_image agent.network_params.architecture=crnn \
env=mazelab \
agent.algo_params.cognition_cost=10,1,0.1 \

