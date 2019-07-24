import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
max_features = 2000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
'''
序列填充
maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0
'''
x_train = sequence.pad_sequences(x_train,maxlen=max_len)
x_test = sequence.pad_sequences(x_test,maxlen=max_len)

model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])
callbacks=[
    TensorBoard(log_dir='./logs',
                 histogram_freq=1,
                 embeddings_freq=1,
                )
]
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=callbacks)