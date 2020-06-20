python train.py -m name=m9_baseline_crnn \
agent=a2c_image agent.network_params.architecture=crnn \
env=mazelab \
seed=0,0,0,0,0 \
agent.algo_params.twoAM=false

