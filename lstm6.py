import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
df = pd.read_csv("protac.csv_processed.csv")

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('Smiles')

# Drop categorical columns from the dataframe
df_numerical = df.drop(columns=categorical_columns + ['Smiles'])

# Apply label encoding to categorical variables
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Combine numerical features with encoded categorical features
features = pd.concat([df_numerical, df[categorical_columns]], axis=1).values.astype(np.float32)

smiles = df['Smiles'].values.astype(str)  # Convert to string type

# Tokenize SMILES using Tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(smiles)

vocab_size = len(tokenizer.word_index) + 1

# Convert SMILES to sequences
sequences = tokenizer.texts_to_sequences(smiles)
max_smiles_length = max(len(seq) for seq in sequences)
y = pad_sequences(sequences, maxlen=max_smiles_length, padding='post')

# One-hot encode the labels
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, y, test_size=0.3, random_state=42)

# Define the model architecture
input_features = layers.Input(shape=(features.shape[1],))
x = layers.Dense(128, activation='relu')(input_features)
x = layers.RepeatVector(max_smiles_length)(x)
x = layers.LSTM(256, return_sequences=True)(x)
x = layers.LSTM(256, return_sequences=True)(x)
output = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(x)

model = tf.keras.Model(inputs=input_features, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model
model.fit(X_train, y_train, epochs=10000, batch_size=64, validation_data=(X_val, y_val))

# Save the model and tokenizer
model.save("smiles_rnn_protac_design_lstm_6_final.h5")
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())
