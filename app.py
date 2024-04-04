import streamlit as st
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision.transforms.functional import to_pil_image

# Define class list
class_list = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define Generator class
class Generator(nn.Module):
    def __init__(self, generator_layer_size, z_size, img_size, class_num):
        super().__init__()

        self.z_size = z_size
        self.img_size = img_size

        self.label_emb = nn.Embedding(class_num, class_num)

        self.model = nn.Sequential(
            nn.Linear(self.z_size + class_num, generator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[2], self.img_size * self.img_size),
            nn.Tanh()
        )

    def forward(self, z, labels):

        # Reshape z
        z = z.view(-1, self.z_size)

        # One-hot vector to embedding vector
        c = self.label_emb(labels)
        print(z.shape,c.shape)
        # Concat image & label
        x = torch.cat([z, c], dim=1)

        # Generator out
        out = self.model(x)
        out = out.view(-1, self.img_size, self.img_size)
        print(out.shape)
        return out

generator_layer_size = [256, 512, 1024]
z_size = 100
img_size = 28
class_num = len(class_list)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator_save_path = "ckpt/generator.pth"

@st.cache_data  # Cache the loaded generator model
def load_generator_model():

    generator = Generator(generator_layer_size, z_size, img_size, class_num).to(device)
    generator.load_state_dict(torch.load(generator_save_path, map_location=device))
    generator.eval()
    return generator

# Function to generate images
def generate_images(class_idx,num_images=6):
    image_list = []
    for _ in range(num_images):
        # Building z (ensure z has batch dimension for multiple images)
        z = Variable(torch.randn(1, z_size)).to(device)

        # Load the cached generator model
        generator = load_generator_model()

        # Generate image
        with torch.no_grad():
            label = Variable(torch.LongTensor([class_idx])).to(device)
            sample_image = generator(z, label).cpu()  # Remove unsqueeze(1) to maintain 3 dimensions
            sample_image_pil = to_pil_image(sample_image)
            image_list.append(sample_image_pil)

    return image_list

class_index = {'T-Shirt':0, 'Trouser':1, 'Pullover':2, 'Dress':3, 'Coat':4, 'Sandal':5, 'Shirt':6, 'Sneaker':7, 'Bag':8, 'Ankle boot':9}

# Streamlit app
def main():
    st.title('CGAN Fashion MNIST Generator')
    
    # Dropdown for selecting class
    selected_class = st.selectbox('Select a class:', class_list)
    # Number of images slider (optional)
    num_images = st.slider('Number of Images to Generate', min_value=1, max_value=10, value=6)

    class_idx = class_list.index(selected_class)
    
    # Generate button
    if st.button('Generate'):
        # Generate image

        generated_images = generate_images(class_idx,num_images)
        cols = st.columns(num_images)
        # Display all generated images in 2 rows and 3 columns
        for i in range(0, num_images, 3):  # Loop in chunks of 3 for each row
            cols = st.columns(3)
            for j in range(3):
                if i + j < num_images:
                    cols[j].image(generated_images[i + j], caption=f"{selected_class} {i+j+1}", width=200)



if __name__ == '__main__':
    main()
