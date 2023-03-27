# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2020 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

import torch


class GECO(object):
    def __init__(self, goal, step_size, alpha=0.95, geco_lambda_init=1,
                 geco_lambda_min=1e-10, geco_lambda_max=1e10, speedup=None):
        self.cma = None
        self.goal = goal
        self.step_size = step_size
        self.alpha = alpha
        self.geco_lambda = torch.tensor(geco_lambda_init)
        self.geco_lambda_min = torch.tensor(geco_lambda_min)
        self.geco_lambda_max = torch.tensor(geco_lambda_max)
        
    def to_cuda(self):
        self.geco_lambda = self.geco_lambda.cuda()
        if self.cma is not None:
            self.cma = self.cma.cuda()

    def to(self, device):
        self.geco_lambda = self.geco_lambda.to(device)
        if self.cma is not None:
            self.cma = self.cma.to(device)

    def state_dict(self):
        return {'cma': self.cma, 'geco_lambda': self.geco_lambda}
        
    def load_state_dict(self, state_dict):
        self.cma = state_dict['cma']
        self.geco_lambda = state_dict['geco_lambda']
            
    def loss(self, err, kld):
        
        constraint = err - self.goal    
        
        if self.cma is None:
            self.cma = constraint
        else:
            self.cma = (1.0 - self.alpha) * constraint + self.alpha * self.cma
            
        with torch.no_grad():
            cons = self.cma
            factor = torch.exp(self.step_size * cons)
            self.geco_lambda = (factor * self.geco_lambda).clamp(self.geco_lambda_min, self.geco_lambda_max)
            
        loss = kld + self.geco_lambda * err

        return loss
