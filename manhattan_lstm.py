import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
from highway import Highway
from char_embedding import CharEmbedding
class Manhattan_LSTM(nn.Module):
    def __init__(self,embedding,args):
        super(Manhattan_LSTM, self).__init__()
        self.embd_size = args.w_embd_size
        self.d = self.embd_size * 2 # word_embedding + char_embedding
        self.use_cuda = torch.cuda.is_available()
        #self.hidden_size = hidden_size
        self.char_embedding = CharEmbedding(args)
        if args.use_embedding:
            self.word_embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.word_embedding.weight = nn.Parameter(embedding)
            #self.input_size = embedding.shape[1] # V - Size of embedding vector

        else:
            self.word_embedding = nn.Embedding(embedding[0], embedding[1])
            #self.input_size = embedding[1]
            
        self.word_embedding.weight.requires_grad = args.train_embedding
        self.W = nn.Linear(6*self.d, 1, bias=False)
        self.highway_net = Highway(self.d)
        #TODO: change the num_layers
        self.modeling_layer = nn.LSTM(8*self.d, self.d, num_layers=1, bidirectional=True, dropout=0.2, batch_first=True)
        self.ctx_embd_layer = nn.LSTM(self.d, self.d, bidirectional = True)
        #self.ctx_embd_layer2 == nn.LSYM(self.d, self.d, bidirectional = True)

        #self.lstm_1 = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=False)
        #self.lstm_2 = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=False)

    def exponent_neg_manhattan_distance(self, x1, x2):
        ''' Helper function for the similarity estimate of the LSTMs outputs '''
        return torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1))

    def forward(self, input):
        '''
        input           -> (2 x Max. Sequence Length (per batch) x Batch Size)
        hidden          -> (2 x Num. Layers * Num. Directions x Batch Size x Hidden Size)
        '''
# =============================================================================
#         embedded_1 = self.embedding(input[0]) # L, B, V
#         embedded_2 = self.embedding(input[1]) # L, B, V
#         #TODO: word & character contextual embedding, namely embd_question1 & embd_question2; define the shape parameter 
# =============================================================================
       #1. word embedding
        word_embedded_1 = self.word_embedding(input[0]).transpose(0,1) # B,L, 300
        word_embedded_2 = self.word_embedding(input[1]).transpose(0,1)# B, L, V

        # 2 char embbedding
        char_embedded_1 = self.char_embedding(input[2]) # B, L, word_len, char_dim
        char_embedded_2 = self.char_embedding(input[3]) # need to convert into max_sent_len, batch_size, char_dim
        
        
        #3 contextual embedding layer
        # Highway network
        embedded_1 = torch.cat((word_embedded_1, char_embedded_1), 2)
        embedded_1 = self.highway_net(embedded_1)

        embedded_2 = torch.cat((word_embedded_2, char_embedded_2), 2)
        embedded_2 = self.highway_net(embedded_2)

        
        embd_question1, h_1 = self.ctx_embd_layer(embedded_1) 
        embd_question2, h_2 = self.ctx_embd_layer(embedded_2)
        
        batch_size = embd_question1.size()[0]
        L = embd_question1.size()[1]
        
        
        # 4. Attention Flow Layer
        # Make a similarity matrix
        shape = (batch_size, L, L, 2*self.d)            # (N, L, L, 2d)
        embd_question1_ex = embd_question1.unsqueeze(2)     # (N, L, 1, 2d)
        embd_question1_ex = embd_question1_ex.expand(shape) # (N, L, L, 2d)
        embd_question2_ex = embd_question2.unsqueeze(1)         # (N, 1, L, 2d)
        embd_question2_ex = embd_question2_ex.expand(shape)     # (N, L, L, 2d)
        a_elmwise_mul_b = torch.mul(embd_question1_ex, embd_question2_ex) # (N, L, L, 2d)
        cat_data = torch.cat((embd_question1_ex, embd_question2_ex, a_elmwise_mul_b), 3) # (N, L, L, 6d), [h;u;h◦u]
        S = self.W(cat_data).view(batch_size, L, L) # (N, L, L)

        # Context2Query
        c2q = torch.bmm(F.softmax(S, dim=-1), embd_question2) # (N, L, 2d) = bmm( (N, L, L), (N, L, 2d) )
        q2c = torch.bmm(F.softmax(S, dim=-1), embd_question1)
        
        
        
        '''
        # Query2Context
        # b: attention weights on the context
        b = F.softmax(torch.max(S, 2)[0], dim=-1) # (N, T)
        q2c = torch.bmm(b.unsqueeze(1), embd_context) # (N, 1, 2d) = bmm( (N, 1, T), (N, T, 2d) )
        q2c = q2c.repeat(1, T, 1) # (N, T, 2d), tiled T times
        '''
        # G: query aware representation of each context word
        G1 = torch.cat((embd_question1, c2q, embd_question1.mul(c2q), embd_question1.mul(q2c)), 2) # (N, L, 8d)
        G2 = torch.cat((embd_question2, q2c, embd_question2.mul(q2c), embd_question2.mul(c2q)), 2) # (N, L, 8d)
        
        
        # 5. Modeling Layer
        M1, _h1 = self.modeling_layer(G1) # M: (N, L, 2d), _h1:(2,B,d)
        M2, _h2 = self.modeling_layer(G2) # M: (N, L, 2d), _h2:(2,B,d)

        def combine_directions(outs):
            return torch.cat([outs[0: outs.size(0): 2], outs[1: outs.size(0): 2]], dim=2)
        hidden_1 = combine_directions(_h1[0]) #(2,B,2d)
        hidden_2 = combine_directions(_h2[0]) #(2,B,2d)
        #print('HMmmmm####',_h1[0].size(),hidden_1.permute(1, 2, 0).size(), batch_size )
        similarity_scores = self.exponent_neg_manhattan_distance(hidden_1.permute(1, 2, 0).view(batch_size, -1),
                                                                 hidden_2.permute(1, 2, 0).view(batch_size, -1))
        return similarity_scores

    def init_weights(self):
        ''' Initialize weights of lstm 1 '''
        for network in [self.W, self.highway_net,self.modeling_layer,self.ctx_embd_layer, self.char_embedding]:  
            for name, param in network.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

# =============================================================================
# =============================================================================
# #         ''' Set weights of lstm 2 identical to lstm 1 '''
# #         #lstm_1 = self.lstm_1.state_dict()
# #         lstm_2 = self.lstm_2.state_dict()
# # =============================================================================
# 
#         for name_1, param_1 in lstm_1.items():
#             # Backwards compatibility for serialized parameters.
#             if isinstance(param_1, torch.nn.Parameter):
#                 param_1 = param_1.data
# 
#             lstm_2[name_1].copy_(param_1)
# =============================================================================
# # =============================================================================
# 
#     def init_hidden(self, batch_size):
#         # Hidden dimensionality : 2 (h_0, c_0) x Num. Layers * Num. Directions x Batch Size x Hidden Size
#         result = torch.zeros(2, 2, batch_size, self.hidden_size)
# 
#         if self.use_cuda: return result.cuda()
#         else: return result
# 
# =============================================================================
