import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model.csnet import ResEncoder, Decoder, AffinityAttention


# Function to visualize the input and output and save them as PNGs
def visualize_output(output, title="Output", channel=0):
    output = output.detach().numpy().squeeze(0)  # Remove batch dimension
    if output.ndim == 3:
        output = output[channel]  # Take the specified channel if multi-channel
    plt.imshow(output, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.savefig(f"{title}.png")  # Save each output as a PNG file
    plt.close()  # Close the figure to prevent overlapping plots

# Function to test encoder, decoder, and affinity attention block
def test_csnet_components():
    # Create dummy input with reduced spatial size: Batch size = 1, Channels = 1, Height = 64, Width = 64
    input_data = torch.randn(1, 1, 64, 64)  # Random input

    # Visualize and save the input image
    print("Visualizing Input Image")
    visualize_output(input_data, title="Input Image", channel=0)  # Visualize and save the input image

    # Initialize Encoder, Decoder, and Affinity Attention Block
    encoder = ResEncoder(in_channels=1, out_channels=32)
    decoder = Decoder(in_channels=32, out_channels=1)
    attention_block = AffinityAttention(in_channels=32)

    # Pass the input through the encoder
    encoded_output = encoder(input_data)
    print("Encoded Output Shape:", encoded_output.shape)
    visualize_output(encoded_output, title="Encoded Output")

    # Pass the encoded output through the Affinity Attention Block
    attention_output = attention_block(encoded_output)
    print("Attention Output Shape:", attention_output.shape)
    visualize_output(attention_output, title="Attention Output")

    # Pass the attention output through the decoder
    decoded_output = decoder(attention_output)
    print("Decoded Output Shape:", decoded_output.shape)
    visualize_output(decoded_output, title="Decoded Output")

# Run the test
test_csnet_components()
