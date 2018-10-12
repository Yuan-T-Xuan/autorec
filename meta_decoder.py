# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from main import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # input should be in shape (1,1,-1)
        print('input:', input.shape)
        print('hidden:', hidden.shape)
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        return self.out(output[0]), hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

hyper_params = [
    ("dim_of_latent_factor", [25, 50, 75, 100, 125, 150]),
    ("l2_reg", [0.1, 0.01, 0.001, 0.0001, 0.00001]),
    ("type_of_interaction", ["PairwiseEuDist", "PairwiseLog", "PointwiseMLPCE"]),
    ("eudist_margin", [0.5, 1.0, 1.5, 2.0]),
    ("mlp_dim1", [-1, 50, 75, 100, 125, 150]),
    ("mlp_dim2", [-1, 50, 75, 100, 125, 150]),
    ("mlp_dim3", [-1, 50, 75, 100, 125, 150])
]

def train(meta_decoder, decoder_optimizer):
  
    decoder_hidden = meta_decoder.initHidden()
    decoder_optimizer.zero_grad()
    
    output = torch.zeros([1, 1, meta_decoder.output_size], device=device)
    softmax = nn.Softmax(dim=1)
    softmax_outputs_stored = list()
    fclayers_for_hyper_params = dict()
    
    for hp in hyper_params:
        fclayers_for_hyper_params[hp[0]] = nn.Linear(meta_decoder.output_size, len(hp[1]))
    print(fclayers_for_hyper_params)
    
    loss = 0
    # check point
    for i in range(3):
        output, decoder_hidden = meta_decoder(output, decoder_hidden)
        print(hyper_params[i])
        print('output:', output.shape)
        softmax_outputs_stored.append(softmax(fclayers_for_hyper_params[hyper_params[i][0]](output)))
        print('softmax_outputs_stored:', softmax_outputs_stored)
        #
    output_interaction = softmax_outputs_stored[-1]
    type_of_interaction = torch.argmax(output_interaction)
    if type_of_interaction == 0:
        # PairwiseEuDist
        for i in range(3, 4):
            output, decoder_hidden = meta_decoder(output, decoder_hidden)
            softmax_outputs_stored.append(softmax(fclayers_for_hyper_params[hyper_params[i][0]](output)))
    elif type_of_interaction == 1:
        # PairwiseLog
        pass
    else:
        # PointwiseMLPCE
        for i in range(3, 6):
            output, decoder_hidden = meta_decoder(output, decoder_hidden)
            softmax_outputs_stored.append(softmax(fclayers_for_hyper_params[hyper_params[i][0]](output)))
    print(len(softmax_outputs_stored))
    
    # 
    resulted_str = []
    for outputs in softmax_outputs_stored:
        print("softmax_outputs: ", outputs)
        idx = torch.argmax(outputs)
        # print('idx:', idx)
        resulted_str.append(idx.item())
    resulted_str = "_".join(map(str, resulted_str))
    print("resulted_str:")
    print(resulted_str)
    
    # 
    reward = calc_reward_given_descriptor(resulted_str)
    print("reward: " + str(reward))
    expectedReward = 0
    for i in range(len(softmax_outputs_stored)):
        logprob = torch.log(torch.max(softmax_outputs_stored[i]))
        expectedReward += logprob * reward
    loss = - expectedReward   
    print('loss:', loss)
    
    # backpropagate the loss according to the policy
    loss.backward()
    decoder_optimizer.step()

def trainIters():
    num_iters = 100
  
    meta_decoder = DecoderRNN(100,100)
    optimizer = optim.Adam(meta_decoder.parameters(), lr=0.01)
    for iteration in range(num_iters):
        train(meta_decoder, optimizer)


trainIters()

