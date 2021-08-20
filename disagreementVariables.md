* Internal Reward: Internal reward defined by the mean over the variance Dynamics Networks loss. Should be single value, but seems to be list/tensor
* External Reward: External Reward defined by the mean over the last slice in buffer.
* 
* recent_best_ext_ret: Recent maximum achieved reward. Taken taken from the info-argument returned by an environment-step
* best_ext_ret: Should be the same value as above. Might differ because both getting assigned in different points of update process. Check whether the values equal
* recent_best_ext_ret=self.rollout.current_max
* best_ext_ret = self.rollout.best_ext_ret
* 
* ev: Explained variance between the Dynamic networks value predictions and the predictions with added advantages
* 
* advmean: Mean over the buffered advantages
* advstd: Standard deviation over the buffered advantages

* ent: Entropy of the stochpol.pd(?). Which part of the network is that? Probably rather the entropy loss of the PPO Agent. Gets scaled with negative entropy coefficicent

* rew_mean: Mean over buffered rewards. The rewards here are the sum of the scaled intrinsic rewards and the scaled and clipped extrinsic rewards
* 
* retmean: The mean over the buffered returns. The returns are the sum of the value networks prediction and the respective advantages.
* retstd: The standard deviation over the buffered returns. The returns are the sum of the value networks prediction and the respective advantages.

* vpredmean: The mean over the value networks prediction.
* vpredstd: The standard deviation over the value networks prediction.

* opt_vf: The recent loss of the value function. Calculated: 0.5 * MSE of value prediction and self.ph_ret
* opt_aux: The chosen feature extractors loss
* opt_pg: The loss of PPOs policy gradient
* opt_clipfrac: Dont really understand the necessity of this one. Probably kick it
* opt-approxkl: Kullback-Leibler-Divergence between the recent actions negative log-probability distribution and the old one. 
* opt_dyn_loss: Partial loss of the dynamics network using the loss of the feature network(?). Ask Viviane for taht one
* opt_tot: Total loss. Sum of entropy loss, pg loss and vf loss
* 
* epcount: Number of episodes. Taken from the info-argument returned by an environment-step
* eplen: Episode-wise length. Taken from the info-argument returned by an environment-step
* n_updates: Number of updates of the PPO network
* ups: Updates per second
* tps: Timesteps per second
* tcount: Total number of timesteps. Taken from the info-argument returned by an environment-step
* total_secs: Total seconds the network is running
  
* rank: The rank of the process in question. 
* eprew: A list of all rewards collected yet.
* eprew_recent : The mean reward collected with the last batch. Might be pretty comparable to my own External Reward logging
