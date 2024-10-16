

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from rdkit import Chem
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Step 1: Data Preprocessing
# Load the CSV file
df = pd.read_csv("protac.csv_processed.csv")

# Encode categorical variables except 'Smiles'
label_encoders = {}
for column in df.columns:
    if column != 'Smiles' and df[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

# Tokenize SMILES strings using RDKit
def tokenize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

df['Smiles'] = df['Smiles'].apply(tokenize_smiles)


# Split data into features and target
X = df.drop(columns=['Smiles'])
y = df['Smiles']

# Convert SMILES strings to sequences (integer encoding)
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, oov_token='UNK')
tokenizer.fit_on_texts(y)
y_sequences = tokenizer.texts_to_sequences(y)

# Padding sequences
max_sequence_length = max(len(seq) for seq in y_sequences)
y_padded_sequences = pad_sequences(y_sequences, maxlen=max_sequence_length, padding='post')

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_padded_sequences, test_size=0.2, random_state=42)

# Print shapes and number of columns before split
print("Shape of X before split:", X.shape)
print("Shape of y before split:", y.shape)
print("Number of columns in X before split:", X.shape[1])
print("Number of columns in y before split:", 1)  # y contains only the 'Smiles' column

# Print shapes and number of columns after split
print("Shape of X_train after split:", X_train.shape)
print("Shape of X_val after split:", X_val.shape)
print("Shape of y_train after split:", y_train.shape)
print("Shape of y_val after split:", y_val.shape)
print("Number of columns in X_train after split:", X_train.shape[1])
print("Number of columns in X_val after split:", X_val.shape[1])
print("Number of columns in y_train after split:", 1)  # y_train contains only the 'Smiles' column
print("Number of columns in y_val after split:", 1)  # y_val contains only the 'Smiles' column

# Number of unique characters in SMILES sequences
num_unique_characters = len(tokenizer.word_index)
print("Number of unique characters in SMILES sequences:", num_unique_characters)

# Maximum sequence length after padding
print("Maximum sequence length after padding:", max_sequence_length)

latent_dim = 100

# Step 2: Define the Generator and Discriminator Models
def build_generator(latent_dim, max_sequence_length):
    model = tf.keras.Sequential([
        layers.Dense(128 * 8 * 8, input_dim=latent_dim, activation='relu'),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # Output layer for SMILES generation
    ])
    return model

# Rebuild the generator with the correct output shape and activation function
generator = build_generator(latent_dim, max_sequence_length)

def build_discriminator(max_sequence_length):
    model = tf.keras.Sequential([
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=[64, 64, 1]),  # Modify input shape here
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)  # Output a single value for binary classification
    ])
    return model


# Rebuild the discriminator with the correct input shape
discriminator = build_discriminator(max_sequence_length)

# Step 3: Define the GAN Model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model

gan = build_gan(generator, discriminator)

# Step 4: Compile the Models and Define Loss Functions
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Step 5: Training
epochs = 100
batch_size = 128

for epoch in range(epochs):
    for _ in range(X_train.shape[0] // batch_size):
        # Train the discriminator
        noise = tf.random.normal([batch_size, latent_dim])
        generated_smiles = generator.predict(noise)
        real_smiles = X_train.iloc[np.random.randint(0, X_train.shape[0], batch_size)]
        X_batch = np.concatenate([real_smiles, generated_smiles], axis=0)
        y_batch = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))], axis=0)
        d_loss = discriminator.train_on_batch(X_batch, y_batch)

        # Train the generator
        noise = tf.random.normal([batch_size, latent_dim])
        y_generated = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, y_generated)

    print(f'Epoch {epoch + 1}, Generator Loss: {g_loss}, Discriminator Loss: {d_loss}')


# Step 6: Generating SMILES Strings
def generate_smiles(generator, latent_dim, num_samples=1):
    noise = tf.random.normal([num_samples, latent_dim])
    generated_smiles = generator(noise)
    return generated_smiles

# Generate SMILES strings
generated_smiles = generate_smiles(generator, latent_dim, num_samples=5)
print("Generated SMILES:", generated_smiles)
