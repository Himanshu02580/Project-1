import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Embedding, Concatenate
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
The Conditional GAN Implementation
Here's the full implementation of our conditional GAN class:

python
class ConditionalGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the label as input
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For combined model, we only train the generator
        self.discriminator.trainable = False

        # The discriminator takes the generated image and label as input
        valid = self.discriminator([img, label])

        # The combined model (stacked generator and discriminator)
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        # Generator input: noise and label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        
        # Embedding layer to convert labels to dense vectors
        label_embedding = Embedding(self.num_classes, 50)(label)
        label_embedding = Flatten()(label_embedding)
        
        # Element-wise multiplication
        model_input = Concatenate()([noise, label_embedding])
        
        # Pass the combined input through the Sequential model
        img = model(Dense(self.latent_dim)(model_input))

        return Model([noise, label], img)

    def build_discriminator(self):
        img = Input(shape=self.img_shape)
        
        # Embedding layer for label input
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Embedding(self.num_classes, np.prod(self.img_shape))(label)
        label_embedding = Flatten()(label_embedding)
        
        # Scale embedding to img size and apply element-wise multiplication
        label_embedding = Dense(np.prod(self.img_shape))(label_embedding)
        label_embedding = Reshape(self.img_shape)(label_embedding)
        
        # Combine image and label
        combined_input = Concatenate(axis=-1)([img, label_embedding])
        
        model = Flatten()(combined_input)
        
        model = Dense(512)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dense(256)(model)
        model = LeakyReLU(alpha=0.2)(model)
        validity = Dense(1, activation='sigmoid')(model)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Load the dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Sample noise and random labels
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            sampled_labels = np.random.randint(0, 10, (batch_size, 1))

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Print the progress
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss}]")

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
    
    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        
        # Create an array of digit labels (0-9)
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        sampled_labels = sampled_labels.reshape(-1, 1)
        
        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title(f"Digit: {sampled_labels[cnt][0]}")
                axs[i,j].axis('off')
                cnt += 1
        
        # Create directory if it doesn't exist
        if not os.path.exists('images'):
            os.makedirs('images')
            
        fig.savefig(f"images/mnist_{epoch}.png")
        plt.close()
        
    def save_model(self):
        # Create directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
            
        self.generator.save('models/mnist_cgan_generator.h5')
        self.discriminator.save('models/mnist_cgan_discriminator.h5')
        
    def generate_specific_digit(self, digit, num_samples=5):
        """Generate specific digits on demand"""
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        labels = np.array([digit] * num_samples).reshape(-1, 1)
        
        gen_imgs = self.generator.predict([noise, labels])
        gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to 0-1
        
        fig, axs = plt.subplots(1, num_samples, figsize=(12, 3))
        for i in range(num_samples):
            axs[i].imshow(gen_imgs[i,:,:,0], cmap='gray')
            axs[i].set_title(f"Generated {digit}")
            axs[i].axis('off')
        plt.show()
Training and Using the Model
Here's how to train the model and generate specific digits:

python
# Initialize the GAN
cgan = ConditionalGAN()

# Train the model
cgan.train(epochs=20000, batch_size=32, sample_interval=1000)

# Save the model
cgan.save_model()

# Generate specific digits
for digit in range(10):
    cgan.generate_specific_digit(digit)
