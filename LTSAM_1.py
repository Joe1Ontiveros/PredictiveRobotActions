#!/usr/python




# This model was used for predictive guessing based on the 
# incoming feed of text gestures
 
 
 
 
 
import torch
import torch.nn as nn


class MiniLTSAM(nn.module):
    # 
    # a CNN model works by 
    # 
    def __init__(self,numc):
        self.cnn = nn.Sequential(
            
        )