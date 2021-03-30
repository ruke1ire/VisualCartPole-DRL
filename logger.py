import os
import numpy as np
import errno
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch
from datetime import datetime

'''
    TensorBoard Data will be stored in './runs' path
'''


class Logger:

    def __init__(self, model_name):
        self.model_name = model_name
        now = datetime.now().strftime("%d/%m %H:%M")
        
        comment = f"{self.model_name} {now}"
        
        logdir = f'runs/{comment}'

        self.writer = SummaryWriter(log_dir=logdir,comment=comment)

    def log_reward(self,episode,reward):
        self.writer.add_scalar('reward', reward, episode)
    
    def save_models(self, generator, discriminator, epoch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(),
                   '{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),
                   '{}/D_epoch_{}'.format(out_dir, epoch))

    def close(self):
        self.writer.close()

