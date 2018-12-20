# MiniProject for solving Lunar Lander problem from Open AI Gym

This project use DDQN to train the agent and passed the game

## Parameter Tuning:  
**python ddnq_vect_train.py -epis XXX** : run XXX episode   
**python ddnq_vect_train.py -memlen XXX** : set replay memory length XXX  
**python ddnq_vect_train.py -gamma XXX** : set discount factor XXX  
**python ddnq_vect_train.py -epsilon XXX** : set epsilon XXX  
**python ddnq_vect_train.py -alpha XXX** : set alpha XXX  
**python ddnq_vect_train.py -batch XXX** : set replay batch size to sample XXX  
**python ddnq_vect_train.py -upfreq XXX** : set target network update frequency XXX  


Those keys can be combined to use like python ddnq_vect_train.py -epis XXX -memlen YYY -gamma ZZZ  
When tunning parameters, the agent will run 1000 episode, output each episode rewards, and output last 200 episode average rewards. (if current episode < 200, then last N episode average until N>=200)  

## Training Mode:  
**python ddnq_vect_train.py -train**  
Use default setting to train agent, until agent meet last 100 episode average bigger than 215. Output lunar_ddq_final.h5  

## Test Model:  
**python ddnq_vect_test.py**  
Read lunar_ddq_final.h5 run test cases. (No replay, fitting and exploration, only use current policy to predict best action)  

## Log files:  
**Parameters Tunning Folder**: contain all tuning log file for plotting purpose  
**Final_train.log** : Applied tuned parameters to train the agent until meet the criterion (last 100 episode average reward > 215)  
**Final_test.log**: Use lunar_ddq_final.h5 to test the model   
**10000_epis.log**: trained 10,000 episode to see the trends of rewards.  
