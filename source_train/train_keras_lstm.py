import tensorflow as tf

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense,Activation, Dropout, Bidirectional
from tensorflow.keras import Sequential

import argparse
import os
import pandas as pd


def RNN():
    model = Sequential()
    layer = model.add(Embedding(80000,128,input_length=500))
    layer = model.add(Bidirectional(LSTM(128)))
    layer = model.add(Dense(128,name='FC1'))
    layer = model.add(Activation('relu'))
    layer = model.add(Dense(1,name='out_layer'))
    layer = model.add(Activation('sigmoid'))
    return model    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--n_epochs', type=int, default=1)

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    val_files = [ os.path.join(args.validation, file) for file in os.listdir(args.validation) ]

    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    train_data = pd.concat(raw_data)

    
    if len(val_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.validation, "validation"))
    val_raw_data = [ pd.read_csv(file, header=None, engine="python") for file in val_files ]
    val_data = pd.concat(val_raw_data)
    
    # labels are in the first column - train data
    train_y = train_data.iloc[:,0]
    train_X = train_data.iloc[:,1:]
    # labels are in the first column - val data
    val_y = val_data.iloc[:,0]
    val_X = val_data.iloc[:,1:]
    
    
    #Build the model
    
    model = RNN()
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    
    #Fit the model
    
    model.fit(train_X,
              train_y,
              batch_size=256,
              epochs=args.n_epochs,
              validation_data=(val_X, val_y))
    
    
    model_path ='/opt/ml/model'
    model.save(os.path.join(model_path,'bi_lstm/1'), save_format='tf')