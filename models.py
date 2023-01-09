from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf



embedding_dim = 128
hidden_units = 256

def bilstm(vocab_size, embedding_dim, hidden_units, tag_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, mask_zero=True))
    
    model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
    model.add(Dropout(.2))
    
    model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
    model.add(Dropout(.2))
    
    
    model.add(TimeDistributed(Dense(tag_size, activation=('softmax'))))
    
    return model