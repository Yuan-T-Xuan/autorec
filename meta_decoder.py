# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from hyper_params import hyper_params
from calc_reward_given_descriptor import calc_reward_given_descriptor

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

moving_average_alpha = 0.2
moving_average = -19013 # a MAGIC NUMBER

class DecoderMLP(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderMLP, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        
        self.dimOutput = nn.Linear(hidden_size, output_size[0])
        self.l2_reg = nn.Linear(hidden_size, output_size[1])
        self.interaction = nn.Linear(hidden_size, output_size[2])
        self.eudist_margin = nn.Linear(hidden_size, output_size[3])
        self.mlp1 = nn.Linear(hidden_size, output_size[4])
        self.mlp2 = nn.Linear(hidden_size, output_size[5])
        self.mlp3 = nn.Linear(hidden_size, output_size[6])


    def forward(self, input):
        out = self.relu(self.fc1(input))
        out = self.relu(self.fc2(out))
        out1 = self.dimOutput(out)
        out2 = self.l2_reg(out)
        out3 = self.interaction(out)
        out4 = self.eudist_margin(out)
        out5 = self.mlp1(out)
        out6 = self.mlp2(out)
        out7 = self.mlp3(out)
        return [out1, out2, out3, out4, out5, out6, out7]


def train(meta_decoder, decoder_optimizer):
    global moving_average
    global moving_average_alpha
    decoder_optimizer.zero_grad()
    input = torch.ones([1, 1], device=device)
    softmax = nn.Softmax(dim=1)
    softmax_outputs_stored = list()
    loss = 0

    output = meta_decoder(input)

    resulted_str = []
    for each in output:
        print("outputs: ", each)
        each_softmax = softmax(each)
        softmax_outputs_stored.append(each_softmax)
        idx = Categorical(each_softmax).sample()
        resulted_str.append(idx.tolist()[0])
    resulted_idx = resulted_str
    resulted_str = "_".join(map(str, resulted_str))
    print("resulted_str: " + resulted_str)
    # 
    reward = calc_reward_given_descriptor(resulted_str)
    if moving_average == -19013:
        moving_average = reward
        reward = 0.0
    else:
        tmp = reward
        reward = reward - moving_average
        moving_average = moving_average_alpha * tmp + (1.0 - moving_average_alpha) * moving_average
    #
    print("current reward: " + str(reward))
    print("current moving average: " + str(moving_average))
    expectedReward = 0
    for i in range(len(softmax_outputs_stored)):
        logprob = torch.log(softmax_outputs_stored[i][0][resulted_idx[i]])
        expectedReward += logprob * reward
    loss = - expectedReward   
    print('loss:', loss)

    # finally, backpropagate the loss according to the policy
    loss.backward()
    decoder_optimizer.step()

def trainIters():
    num_iters = 1000
  
    meta_decoder = DecoderMLP(100,[6,5,3,4,6,6,6])

    optimizer = optim.SGD(meta_decoder.parameters(), lr=0.001, momentum=0.9)
    for iteration in range(num_iters):
        train(meta_decoder, optimizer)


if __name__ == "__main__":
    trainIters()

