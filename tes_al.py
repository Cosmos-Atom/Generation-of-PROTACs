import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the saved model
model = tf.keras.models.load_model("smiles_rnn_protac_design.h5")

# Load and preprocess the data
df = pd.read_csv("albendazole.csv")

# Preprocess the SMILES column without modification
tokenizer = Tokenizer(char_level=True, filters='')
tokenizer.fit_on_texts(df['Smiles'])
vocab_size = len(tokenizer.word_index) + 1

# Function to generate new SMILES strings
def generate_smiles(model, tokenizer, max_smiles_length):
    # Random input vector, for instance, as a placeholder
    random_input = np.random.randint(1, vocab_size, size=(1, max_smiles_length))
    
    generated_smiles = model.predict(random_input)
    generated_smiles = np.argmax(generated_smiles, axis=-1)
    generated_smiles = generated_smiles[0]  # Get the first sequence (as batch size is 1)
    
    decoded_generated_smiles = ''.join(tokenizer.index_word[idx] for idx in generated_smiles if idx != 0)
    
    return decoded_generated_smiles

# Prepare to generate SMILES without using original SMILES
max_smiles_length = max(len(seq) for seq in tokenizer.texts_to_sequences(df['Smiles']))

# Generate SMILES for Albendazole
results = []

for _ in range(len(df)):
    generated_smiles = generate_smiles(model, tokenizer, max_smiles_length)
    results.append(generated_smiles)

# Print results
for generated in results:
    print(f'Generated SMILES: {generated}')
    print('---')
