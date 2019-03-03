import random

import torch
import torch.nn.utils.rnn as rnn
import numpy as np

class Train_Network(object):
    def __init__(self, manhattan_lstm, index2word):
        self.manhattan_lstm = manhattan_lstm
        self.index2word = index2word
        self.use_cuda = torch.cuda.is_available()

    def train(self, input_sequences, input_char, similarity_scores, criterion, model_optimizer=None, evaluate=False):
        #print('befor batch input sequences')
        #print('00000009909090900380808038408048038')
        #print(input_char) 
        for i, pair in enumerate(input_char):
            for j, pairs in enumerate(input_char[i]):
                input_char[i][j] = torch.FloatTensor(input_char[i][j])
                #print(type(type(input_char[i][j])))
            #if type(input_char[i]) == 
        sequences_1 = [sequence[0] for sequence in input_sequences]
        #print('sequences_1, after batch ')
        #print(sequences_1)
        sequences_2 = [sequence[1] for sequence in input_sequences]
        #print(sequences_1)
        
        batch_size = len(sequences_1) 
        def process_batch_char(a):    
            sen_long = []
            char_long = []
            for i in a:
                #print('&&&&&&&&&&&&&&&')
                #print(i)
                for j in i:
                    #print('@@@@@@@@@@@')
                    #print(j)
                    #print(j.size())
                    sen_long.append(j.size()[0])
                    char_long.append(j.size()[1])
            sen_length = max(sen_long)
            char_length = max(char_long)
            for i, pair in enumerate(a):
                for j, _ in enumerate(a[i]):
                    sen_l = sen_length - len(a[i][j])
                    char_l = char_length - a[i][j].size()[1]
                    if sen_l > 0:
                        e = torch.zeros((sen_l, a[i][j].size()[1]))
                        a[i][j] = torch.cat((a[i][j], e), 0)
                    if char_l > 0:
                        e = torch.zeros((sen_length, char_l))
                        a[i][j] = torch.cat((a[i][j], e), 1)
                    a[i][j] = np.array(a[i][j])
            return torch.LongTensor(a)
        input_char = process_batch_char(input_char)
        #print(input_char)
        sequences_1_char = []
        sequences_2_char = []
        #print('mamamadsmsdkfshfsfashf;sdfhsdhf;sadh')
        #print(input_char)
        for i in input_char:
            sequences_1_char.append(np.array(i[0]))
        #print('mamamadsmsdkfshfsfashf;sdfhsdhf;sadhdafsdfasdf;kslfjasjfsalf;laslf')
        #print(input_char)
        sequences_1_char = torch.LongTensor(sequences_1_char)
    
        for i in input_char:
            sequences_2_char.append(np.array(i[1]))
        sequences_2_char = torch.LongTensor(sequences_2_char)
 

        '''
        Pad all tensors in this batch to same length.
        PyTorch pad_sequence method doesn't take pad length, making this step problematic.
        Therefore, lists concatenated, padded to common length, and then split.
        '''
        #print('-----------------')
        
        #print(sequences_1)
        #print('*****************')
        #print(sequences_2)
        temp = rnn.pad_sequence(sequences_1 + sequences_2)
        #print(temp)
        #temp1 = rnn.pad_sequence(sequences_1_char + sequences_2_char)
        
        
        sequences_1 = temp[:, :batch_size]
        sequences_2 = temp[:, batch_size:] 
        #print(sequences_1)
        #print(sequences_1.size())
        #print(sequences_2)

        ''' No need to send optimizer in case of evaluation. '''
        if model_optimizer: model_optimizer.zero_grad()
        loss = 0.0

        #hidden = self.manhattan_lstm.init_hidden(batch_size)
        ###########################################################
        #把charcter embedding 层加到manhattan——lstm里，然后在这传入，sequences_1_char, sequences_2_char
        output_scores = self.manhattan_lstm([sequences_1, sequences_2,sequences_1_char,sequences_2_char]).view(-1)
        #####val 返回loss要先改manhattan——lstm 然后最后再处理val的chra 输入
        loss += criterion(output_scores, similarity_scores)

        if not evaluate:
            loss.backward()
            model_optimizer.step()

        return loss.item(), output_scores
