python train.py agent=a2c_crnn agent.network.recurse_depth=0
python train.py agent=a2c_crnn agent.network.recurse_depth=1
python train.py agent=a2c_crnn agent.network.recurse_depth=2
python train.py agent=a2c_2am
python train.py agent=a2c_2am agent.long_horizon=true
python train.py agent=a2c_2am agent.long_horizon=true agent.cognition_cost=0.001