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

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # input should be in shape (1,1,-1)
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        return self.out(output[0]), hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def train(meta_decoder, decoder_optimizer, fclayers_for_hyper_params):
    global moving_average
    global moving_average_alpha
    decoder_hidden = meta_decoder.initHidden()
    decoder_optimizer.zero_grad()
    output = torch.zeros([1, 1, meta_decoder.output_size], device=device)
    softmax = nn.Softmax(dim=1)
    softmax_outputs_stored = list()
    loss = 0
    # 
    for i in range(3):
        output, decoder_hidden = meta_decoder(output, decoder_hidden)
        #print(hyper_params[i])
        softmax_outputs_stored.append(softmax(fclayers_for_hyper_params[hyper_params[i][0]](output)))
        #
    output_interaction = softmax_outputs_stored[-1]
    type_of_interaction = Categorical(output_interaction).sample().tolist()[0]
    if type_of_interaction == 0:
        # PairwiseEuDist
        for i in range(3, 4):
            output, decoder_hidden = meta_decoder(output, decoder_hidden)
            softmax_outputs_stored.append(softmax(fclayers_for_hyper_params[hyper_params[i][0]](output)))
    elif type_of_interaction == 1:
        # PairwiseLog
        # no hyper-params for this interaction type
        pass
    else:
        # PointwiseMLPCE
        for i in range(4, 7):
            output, decoder_hidden = meta_decoder(output, decoder_hidden)
            softmax_outputs_stored.append(softmax(fclayers_for_hyper_params[hyper_params[i][0]](output)))
    # 
    resulted_str = []
    for outputs in softmax_outputs_stored:
        print("softmax_outputs: ", outputs)
        idx = Categorical(outputs).sample()
        resulted_str.append(idx.tolist()[0])
    resulted_str[2] = type_of_interaction   # the type of interaction has already been sampled before
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
  
    meta_decoder = DecoderRNN(100,100)
    fclayers_for_hyper_params = dict()
    
    for hp in hyper_params:
        fclayers_for_hyper_params[hp[0]] = nn.Linear(meta_decoder.output_size, len(hp[1]))
    print(fclayers_for_hyper_params)

    optimizer = optim.Adam(meta_decoder.parameters(), lr=0.01)
    for iteration in range(num_iters):
        train(meta_decoder, optimizer, fclayers_for_hyper_params)


if __name__ == "__main__":
    trainIters()

