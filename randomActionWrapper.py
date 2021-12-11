import random 
import gym

class RandomActionWrapper(gym.ActionWrapper):
    
    def action(self, action):
<<<<<<< HEAD
        actions = [0,1,2,3,4,5]
=======
        actions = list(self.actions)
>>>>>>> 6fb1b42467ca0ab4cac12919fabf72982865364a
        randomNumber = random.randint(0,9)
        randomWhen = [3,5,6]
        if randomNumber in randomWhen:
            action = random.choice(actions)
        return action

