############ ENVIRONMENT ##############

import gym
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

RESIZE_PIXELS = 40

class Environment:
    def __init__(self):
        self.env = gym.make("CartPole-v0").unwrapped
        self.env.reset()
        
        screen = self.env.render(mode='rgb_array').transpose((2,0,1))
        _, self.screen_height, self.screen_width = screen.shape
        
        self.resize = T.Compose([T.ToPILImage(),
                    T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
                    T.Grayscale(),
                    T.ToTensor()])
        
        world_width = self.env.x_threshold * 2
        self.scale = self.screen_width / world_width
        
    def get_cart_location(self):
        return int(self.env.state[0] * self.scale + self.screen_width / 2.0)
        
    def get_screen(self):
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        
        screen = screen[:, int(self.screen_height*0.4):int(self.screen_height * 0.8)]
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        return self.resize(screen).unsqueeze(0)
    
    def close(self):
        self.env.close()
        
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        self.env.reset()
