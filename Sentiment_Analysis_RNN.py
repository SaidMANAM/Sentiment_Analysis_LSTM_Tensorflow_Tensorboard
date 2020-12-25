import time
import os
import nltk
import datetime
import numpy as np
import pandas as pd
import tensorflow
import tensorflow_hub as hub
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Bidirectional, Embedding, Dense, SpatialDropout1D, Dropout, GRU
from keras.models import Sequential
from keras.callbacks import TensorBoard
from tensorboard.plugins import projector

start = time.time()

# importing data

Train = pd.read_csv(r'C:\Users\Utilisateur\Downloads\imdb\Train.csv', encoding='UTF-8')
Test = pd.read_csv(r'C:\Users\Utilisateur\Downloads\imdb\Test.csv', encoding='UTF-8')
Valid = pd.read_csv(r'C:\Users\Utilisateur\Downloads\imdb\Valid.csv', encoding='UTF-8')

# download stopwords

# nltk.download()
# nltk.download('stopwords')
stemmer = SnowballStemmer("english")
lem = WordNetLemmatizer()


def Split(data):
    v = data['label']
    w = data.drop('label', axis=1)
    return v, w


def val(data):
    v = data['text'].values
    return v


def va(data):
    v = data.values
    return v


def Process(data):
    sw = stopwords.words('english')
    data['text'] = data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    data['text'] = data['text'].apply(lambda x: lem.lemmatize(x))


Y_train, X_train = Split(Train)
Y_test, X_test = Split(Test)
Y_valid, X_valid = Split(Valid)

Process(X_train)
Process(X_test)
Process(X_valid)

X_tr = val(X_train)
Y_tr = va(Y_train)
X_te = val(X_test)
Y_te = va(Y_test)
X_va = val(X_valid)
Y_va = va(Y_valid)
vocab = np.concatenate((X_tr, X_va))

# Model's parameters #

vocab_size = 40000
embedding_dimension = 64
max_length = 120
trunc = 'post'
oov_tok = '<OOV>'
epochs = 2

# Turning our inputs into sequences #

tokenizer = Tokenizer(num_words=vocab_size,
                      filters='''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~¡¢£¤¦§¨«­®°³´·º»½¾¿ßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþğıōżאגויכלמןר–‘’“”…″₤★、''',
                      oov_token=oov_tok)
tokenizer.fit_on_texts(vocab)
X = tokenizer.texts_to_sequences(X_tr)
sequences = tokenizer.texts_to_sequences(X_tr)
X_padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating=trunc)
X_val_seq = tokenizer.texts_to_sequences(X_va)
X_val_padded = pad_sequences(X_val_seq, maxlen=max_length, padding='post', truncating=trunc)
X_test_seq = tokenizer.texts_to_sequences(X_te)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating=trunc)
word_index = tokenizer.word_index

# The model #

model = Sequential()
model.add(Embedding(vocab_size, embedding_dimension, input_length=max_length))
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(LSTM(120, activation='tanh', return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(120, activation='tanh', return_sequences=False)))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])
log_dir = r'C:\Users\Utilisateur\PycharmProjects\pythonProject2' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(X_padded, Y_train, epochs=epochs, batch_size=16, validation_data=(X_val_padded, Y_va),
                    callbacks=[tensorboard_callback])
vy = history.history['val_loss']
ty = history.history['val_accuracy']
model.summary()
with open(os.path.join(log_dir, 'metadata.tsv'), "w", encoding="utf-8") as f:
    for subwords in word_index.keys():
        if subwords != '<OOV>':
            f.write("{}\n".format(subwords))

weights = tensorflow.Variable(model.layers[0].get_weights()[0][1:])
# Create a checkpoint from embedding, the filename and key are
# name of the tensor.
checkpoint = tensorflow.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = os.path.join(log_dir, 'metadata.tsv')
projector.visualize_embeddings(log_dir, config)


def Get_Sentiment(x):
    test = [x]
    test_seq = tokenizer.texts_to_sequences(test)
    test_padded = pad_sequences(test_seq, maxlen=max_length, padding='post', truncating=trunc)
    if (model.predict(test_padded) > 0.5).astype("int32"):
        print("Positive", (model.predict(test_padded)))
    else:
        print("Negative", (model.predict(test_padded)))


review = str(input("Enter the review :  "))
Get_Sentiment(review)
print('val_loss', vy)
print('val_accuracy', ty)

res = model.evaluate(X_test_padded, Y_te)

print('test evaluation: ', res)
end = time.time()
print(f"Runtime of the program is {end - start}")
