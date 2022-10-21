# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.linear = nn.Linear(768, 768)

    def forward(self, code_inputs, nl_inputs, return_vec=False):
        bs=code_inputs.shape[0]
        inputs=torch.cat((code_inputs,nl_inputs),0)
        model_outputs=self.encoder(inputs,attention_mask=inputs.ne(1))
        outputs = model_outputs[0]

        code_vec = torch.sum(outputs[:bs, :2, :], dim=1)
        nl_vec = torch.sum(outputs[bs:, :2, :], dim=1)
        code_vec = self.linear(code_vec)
        nl_vec = self.linear(nl_vec)

        code_vec = torch.nn.functional.normalize(code_vec, p=2, dim=1)
        nl_vec = torch.nn.functional.normalize(nl_vec, p=2, dim=1)

        if return_vec:
            return code_vec,nl_vec
        scores=(nl_vec[:,None,:]*code_vec[None,:,:]).sum(-1)/0.05
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        return loss, code_vec, nl_vec
