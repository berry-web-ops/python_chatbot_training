import numpy as np
import tensorflow as tf
from preprocess import processed_data_questions, processed_data_answers
 
oov_tok = '<OOV>'
trunc_type='post'
padding_type='post'
max_length = 100
embedding_dim = 3
training_labels_size = len(processed_data_answers)
training_data_size = len(processed_data_questions)

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_tok)
tokenizer.fit_on_texts(processed_data_questions)
sequences = tokenizer.texts_to_sequences(processed_data_questions)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_data = padded_sequences[:training_data_size]
training_labels = padded_sequences[:training_labels_size]
vocab_size = len(tokenizer.word_index) + 1

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(training_data_size, embedding_dim),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(max_length, activation='softmax')
])
#training_size_labels
#sparse_categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#50
num_epochs = 2
history = model.fit(training_data, training_labels, epochs=num_epochs, verbose=2)