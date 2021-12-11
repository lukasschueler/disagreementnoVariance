import gym
import seaborn as sns
import wandb
import numpy as np

class stateCoverage(gym.core.Wrapper):

<<<<<<< HEAD
    def __init__(self, env, envSize=8, recordWhen=10):
=======
    def __init__(self, env, envSize, recordWhen, rank):
>>>>>>> 6fb1b42467ca0ab4cac12919fabf72982865364a
        super().__init__(env)
        self.envSize = envSize
        self.counts = {}
        self.numberTimesteps = 0
        self.recordWhen = recordWhen
<<<<<<< HEAD
        
=======
        self.rank = rank
>>>>>>> 6fb1b42467ca0ab4cac12919fabf72982865364a
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.rank == 0:
            self.numberTimesteps += 1

<<<<<<< HEAD
        if action == 2:
=======
            # Tuple based on which we index the counts
            # We use the position after an update
>>>>>>> 6fb1b42467ca0ab4cac12919fabf72982865364a
            env = self.unwrapped
            tup = (tuple(env.agent_pos))

            # Get the count for this key
            pre_count = 0
            if tup in self.counts:
                pre_count = self.counts[tup]

            # Update the count for this key
            new_count = pre_count + 1
            self.counts[tup] = new_count

<<<<<<< HEAD
        if self.numberTimesteps % self.recordWhen == 0:
            grid = np.zeros((self.envSize, self.envSize))
            for key, value in self.counts.items():
                x = key[0]
                y = key[1]
                grid[y][x] = value
            grid_cropped = grid[1:-1,1:-1]
            svm= sns.heatmap(grid_cropped)
            # figure = svm.get_figure()    
            # figure.savefig('svm_conf.png', dpi=400)
            wandb.log({"Coverage": [wandb.Image(svm)]})
=======
            if self.numberTimesteps % self.recordWhen == 0:
                self.createHeatmap(self.counts, self.envSize)
>>>>>>> 6fb1b42467ca0ab4cac12919fabf72982865364a

        return obs, reward, done, info

    def reset(self, **kwargs):
<<<<<<< HEAD
        return self.env.reset(**kwargs)
=======
        return self.env.reset(**kwargs)
    
    def createHeatmap(self, dictionary, envSize):
        grid = np.zeros((envSize, envSize))
        for key, value in dictionary.items():
            x = key[0]
            y = key[1]
            grid[x-1][y-1] = value
        heatmap = sns.heatmap(grid)
        wandb.log({"Coverage": [wandb.Image(heatmap)]})
>>>>>>> 6fb1b42467ca0ab4cac12919fabf72982865364a
