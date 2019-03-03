import argparse

import torch

from data import Data
from embedding_google import Get_Embedding
from manhattan_lstm import Manhattan_LSTM
from train_network import Train_Network
from run_iterations import Run_Iterations

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-dn", "--data_name", type=str, help="Dataset name.", default="quora")
    parser.add_argument("-df", "--data_file", type=str, help="Path to dataset.", default="../Datasets/quora.tsv")
    parser.add_argument("-e", "--embd_file", type=str, help="Path to Embedding File.", default="GoogleNews-vectors-negative300.bin.gz")
    parser.add_argument("-tr", "--training_ratio", type=float, help="Ratio of training set.", default=0.8)
    parser.add_argument("-l", "--max_len", type=int, help="Maximum number of words in a sentence.", default=20)
    parser.add_argument("-tp", "--tracking_pair", type=bool, help="Track change in outputs over a randomly chosen sample.", default=False)
#    parser.add_argument("-z", "--hidden_size", type=int, help="Number of Units in LSTM layer.", default=50)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size.", default=50)
    parser.add_argument('-filter', '--filters',type=list, help='filter size', default=[[1,5]])
    parser.add_argument('-out_chs', '--out_chs',type=int, help='character embedding output size', default=300)
    parser.add_argument("-n", "--num_iters", type=int, help="Number of iterations/epochs.", default=100)
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate for optimizer.", default=0.0001) 
    parser.add_argument("-wd", "--w_embd_size", type=int, help="embedding dim for the word", default=300)
    parser.add_argument("-cd", "--c_embd_size", type=int, help="embedding dim for the character.", default=8)
    parser.add_argument("-vc", "--vocab_size_c", type=int, help="vocabulary size of the character", default=44)
    parser.add_argument("-ue", "--use_embedding", type=bool, help="Whether using word embedding", default=True)
    parser.add_argument("-te", "--train_embedding", type=int, help="Training embedding", default=True)
    args = parser.parse_args()

    print('Model Parameters:')
    #print('Hidden Size                  :', args.hidden_size)
    print('Batch Size                   :', args.batch_size)
    print('Max. input length            :', args.max_len)
    print('Learning rate                :', args.learning_rate)
    print('Number of Epochs             :', args.num_iters)
    print('--------------------------------------\n')

    print('Reading Data.')
    data = Data(args.data_name, args.data_file, args.training_ratio, args.max_len)

    print('\n')
    print('Number of training samples        :', len(data.x_train))
    print('Number of validation samples      :', len(data.x_val))
    print('Maximum sequence length           :', args.max_len)
    print('\n')

    print('Building Embedding Matrix')
    embedding = Get_Embedding(args.embd_file, data.word2index,args)
    embedding_size = embedding.embedding_matrix.shape[1]

    print('Building model.')
    model = Manhattan_LSTM(embedding.embedding_matrix, args)
    if use_cuda: model = model.cuda()

    model.init_weights() # set weights to be same in lstm1 and lstm2

    print("Training Network.")
    train_network = Train_Network(model, data.index2word)
    run_iterations = Run_Iterations(train_network, data.x_train, data.x_train_char, data.y_train, data.index2word,
                                    args.batch_size, args.num_iters, args.learning_rate,
                                    tracking_pair=args.tracking_pair, x_val=data.x_val, x_val_char = data.x_val_char, y_val=data.y_val)
    run_iterations.train_iters()
    #run_iterations.evaluate_randomly()

    torch.save(model.state_dict(), './manhattan_lstm.pt')
