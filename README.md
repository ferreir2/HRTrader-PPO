# HRTrader-PPO

Template to train a Horse Racing online trader agent using a Proximal Policy Optimization (PPO) Reinforcement Learning (RL) algorithm.

The implementation only allows for one "open trade" at a time.
For example, if the agent has placed a lay bet and then backs, the back bet size is adjusted so that the book is greened up.
It can however place multiple consecutive bets on the same side.
