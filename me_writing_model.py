#%%
import tensorflow as tf
import numpy as np
import keras

#%%
#Loading and reading text file
writing = open(r"C:\Users\Owner\OneDrive\Desktop\CS Projects\my_writing_bot\data\writing_by_me.txt").read()

#%%
#Encoding the text file into int values
vocab = sorted(set(writing))
char_to_id = {u:i for i, u in enumerate(vocab)}
id_to_char = np.array(vocab)

def text_to_id(txt):
    return np.array([char_to_id[i] for i in txt])

writing_as_id = text_to_id(writing)
print(writing_as_id)

def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(id_to_char[ints])

# %%
#Creating Training Examples
seq_len = 100
examples_per_epoch = len(writing) // (seq_len + 1) #+1 because we want to train the model to predict the last character in each example

char_dataset = tf.data.Dataset.from_tensor_slices(writing_as_id)
sequences = char_dataset.batch(seq_len+1, drop_remainder=True)

def split_input_target(example):
    input = example[:-1]
    target = example[1:]
    return input, target

dataset = sequences.map(split_input_target)

# %%
#Creating Batches from Data
batch_size = 32
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

buffer_size = 10000

data = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

#%%
#Building the Model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
model.summary()

# %%
#Creating a loss function
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

#%%
#Compiling the model
model.compile(optimizer='adam', loss=loss)

#%%
#Training the model
history = model.fit(data, epochs=1)

#%%
#Generating Text
def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 800

  # Converting our start string to numbers (vectorizing)
  input_eval = [char_to_id[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      print(predictions)
      # remove the batch dimension

      predictions = tf.squeeze(predictions)
      print(predictions)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(id_to_char[predicted_id])

  return (start_string + ''.join(text_generated))

inp = "Ellie is cool and sick and awesome"
print(generate_text(model, inp))

# %%
