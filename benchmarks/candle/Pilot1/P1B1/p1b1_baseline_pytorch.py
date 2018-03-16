from __future__ import print_function

import numpy as np

import argparse

import torch
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import p1b1
import p1_common
import p1_common_pytorch


def get_p1b1_parser():

	parser = argparse.ArgumentParser(prog='p1b1_baseline', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Train Autoencoder - Pilot 1 Benchmark 1')

	return p1b1.common_parser(parser)


def main():

    # Get command-line parameters
    parser = get_p1b1_parser()
    args = parser.parse_args()
    #print('Args:', args)
    # Get parameters from configuration file
    fileParameters = p1b1.read_config_file(args.config_file)
    #print ('Params:', fileParameters)
    # Consolidate parameter set. Command-line parameters overwrite file configuration
    gParameters = p1_common.args_overwrite_config(args, fileParameters)
    print ('Params:', gParameters)

    # Construct extension to save model
    ext = p1b1.extension_from_parameters(gParameters, '.pt')
    logfile = args.logfile if args.logfile else args.save+ext+'.log'
    p1b1.logger.info('Params: {}'.format(gParameters))

    # Get default parameters for initialization and optimizer functions
    kerasDefaults = p1_common.keras_default_config()
    seed = gParameters['rng_seed']

    # Load dataset
    X_train, X_val, X_test = p1b1.load_data(gParameters, seed)
    
    print ("Shape X_train: ", X_train.shape)
    print ("Shape X_val: ", X_val.shape)
    print ("Shape X_test: ", X_test.shape)

    print ("Range X_train --> Min: ", np.min(X_train), ", max: ", np.max(X_train))
    print ("Range X_val --> Min: ", np.min(X_val), ", max: ", np.max(X_val))
    print ("Range X_test --> Min: ", np.min(X_test), ", max: ", np.max(X_test))


    # Set input and target to X_train
    train_data = torch.from_numpy(X_train)
    train_tensor = data.TensorDataset(train_data, train_data)
    train_iter = data.DataLoader(train_tensor, batch_size=gParameters['batch_size'], shuffle=gParameters['shuffle'])

    # Validation set
    val_data = torch.from_numpy(X_val)
    val_tensor = torch.utils.data.TensorDataset(val_data, val_data)
    val_iter = torch.utils.data.DataLoader(val_tensor, batch_size=gParameters['batch_size'], shuffle=gParameters['shuffle'])

    # Test set
    test_data = torch.from_numpy(X_test)
    test_tensor = torch.utils.data.TensorDataset(test_data, test_data)
    test_iter = torch.utils.data.DataLoader(test_tensor, batch_size=gParameters['batch_size'], shuffle=gParameters['shuffle'])
    
    #net = mx.sym.Variable('data')
    #out = mx.sym.Variable('softmax_label')
    input_dim = X_train.shape[1]
    output_dim = input_dim

    # Define Autoencoder architecture
    layers = gParameters['dense']
    activation = p1_common_pytorch.build_activation(gParameters['activation'])
    loss_fn = p1_common_pytorch.get_function(gParameters['loss'])
    
    '''
    N1 = layers[0]
    NE = layers[1]

    net = nn.Sequential(
      nn.Linear(input_dim,N1),
      activation,
      nn.Linear(N1,NE),
      activation,
      nn.Linear(NE,N1),
      activation,
      nn.Linear(N1,output_dim),
      activation,
    )
    '''

    # Documentation indicates this should work
    net = nn.Sequential()
    
    if layers != None:
        if type(layers) != list:
            layers = list(layers)
        # Encoder Part
        for i,l in enumerate(layers):
            if i==0:
               net.add_module('in_dense', nn.Linear(input_dim,l))
               net.add_module('in_act', activation)
               insize=l
            else:
               net.add_module('en_dense%d' % i, nn.Linear(insize,l))
               net.add_module('en_act%d' % i, activation)
               insize=l

        # Decoder Part
        for i,l in reversed( list(enumerate(layers)) ):
            if i < len(layers)-1:
               net.add_module('de_dense%d' % i, nn.Linear(insize,l))
               net.add_module('de_act%d' % i, activation)
               insize=l
                    
    net.add_module('out_dense', nn.Linear(insize,output_dim))
    net.add_module('out_act', activation)

    # Initialize weights 
    for m in net.modules():
        if isinstance(m,nn.Linear):
            p1_common_pytorch.build_initializer(m.weight, gParameters['initialization'], kerasDefaults)
            p1_common_pytorch.build_initializer(m.bias, 'constant', kerasDefaults, 0.0)

    # Display model
    print(net)

    # Define context

    # Define optimizer
    optimizer = p1_common_pytorch.build_optimizer(net, gParameters['optimizer'], 
                                                gParameters['learning_rate'],
                                                kerasDefaults)

    # Seed random generator for training
    torch.manual_seed(seed)

    #use_gpu = torch.cuda.is_available()
    use_gpu=0

    train_loss=0

    freq_log = 1
    for epoch in range(gParameters['epochs']):
        for batch, (in_train, _) in enumerate(train_iter):
            in_train = Variable(in_train)
            #print(in_train.data.shape())
            if use_gpu:
                in_train=in_train.cuda()
            optimizer.zero_grad()
            output = net(in_train)
    
            loss = loss_fn(output, in_train)
            loss.backward()
            train_loss +=loss.data[0]
            optimizer.step()
	    if batch % freq_log == 0: 
	        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
	                    epoch, batch * len(in_train), len(train_iter.dataset),
			                    100. * batch / len(train_iter),
					    loss.data[0]))# / len(in_train)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
	        epoch, train_loss / len(train_iter.dataset)))

        # model save
        #save_filepath = "model_ae_" + ext
        #ae.save(save_filepath)

        # Evalute model on valdation set
        for i, (in_val, _) in enumerate(val_iter):
            in_val = Variable(in_val)
            X_pred = net(in_val).data.numpy()
	    if i==0 :
	        in_all = in_val.data.numpy()
	        out_all = X_pred
	    else:
	        in_all = np.append(in_all, in_val.data.numpy(),axis=0)
	        out_all = np.append(out_all, X_pred,axis=0)

        #print ("Shape in_all: ", in_all.shape)
        #print ("Shape out_all: ", out_all.shape)
    
        scores = p1b1.evaluate_autoencoder(in_all, out_all)
        print('Evaluation on validation data:', scores)

    # Evalute model on test set
    for i, (in_test, _) in enumerate(test_iter):
        in_test = Variable(in_test)
        X_pred = net(in_test).data.numpy()
	if i==0 :
	    in_all = in_test.data.numpy()
	    out_all = X_pred
	else:
	    in_all = np.append(in_all, in_test.data.numpy(),axis=0)
	    out_all = np.append(out_all, X_pred,axis=0)

    #print ("Shape in_all: ", in_all.shape)
    #print ("Shape out_all: ", out_all.shape)
    
    scores = p1b1.evaluate_autoencoder(in_all, out_all)
    print('Evaluation on test data:', scores)

    diff = in_all-out_all
    plt.hist(diff.ravel(), bins='auto')
    plt.title("Histogram of Errors with 'auto' bins")
    plt.savefig('histogram_mx.pdf')


if __name__ == '__main__':
    main()
