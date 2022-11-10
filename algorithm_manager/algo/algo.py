from keras.models import Sequential
from keras.layers import SimpleRNN,LSTM, GRU,Dense, Activation


def lstm_model(input_shape,loss,optimizer):
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(32,input_shape=input_shape, return_sequences=True))
    print(model.layers)
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(1,activation='relu'))
    model.compile(loss=loss, optimizer=optimizer)
    return model


def gru_model(input_shape,loss,optimizer):
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(GRU(32,input_shape=input_shape, return_sequences=True))
    print(model.layers)
    model.add(GRU(64, return_sequences=False))
    model.add(Dense(1,activation='relu'))
    model.compile(loss=loss, optimizer=optimizer)
    return model


def rnn_model(input_shape,loss,optimizer):
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(SimpleRNN(32,input_shape=input_shape, return_sequences=True))
    print(model.layers)
    model.add(SimpleRNN(64, return_sequences=False))
    model.add(Dense(1,activation='relu'))
    model.compile(loss=loss, optimizer=optimizer)
    return model