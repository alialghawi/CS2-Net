import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

from utils.misc import thresh_OTSU
from utils.evaluation_metrics import AccSenSpe
device = torch.device("cpu")
# Update the args for a single image
args = {
    'test_image': './21_training.tif',  # Single test image
    'pred_path' : 'assets/DRIVE/',      # Directory to save the prediction
    'img_size'  : 512                   # Resize the image to 512x512
}

# Create the prediction directory if it doesn't exist
if not os.path.exists(args['pred_path']):
    os.makedirs(args['pred_path'])


def load_net():
    """Load the pre-trained model"""
    net = torch.load('./checkpoint/CS_Net_single_image_100.pkl')  # Replace with your actual model checkpoint
    return net


def save_prediction(pred, filename=''):
    """Save the prediction as an image"""
    save_path = args['pred_path'] + 'pred/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Directory created for saving predictions!")

    # Convert the predicted tensor to a numpy array and save it as an image
    mask = pred.data.cpu().numpy() * 255
    mask = np.transpose(np.squeeze(mask, axis=0), [1, 2, 0])
    mask = np.squeeze(mask, axis=-1)

    # Save the mask
    pred_filename = os.path.join(save_path, filename + '.png')
    Image.fromarray(mask.astype(np.uint8)).save(pred_filename)


def predict():
    """Run the prediction on the single test image"""
    net = load_net()
    net.eval()

    # Load and preprocess the single image
    image = Image.open(args['test_image'])
    image = image.resize((args['img_size'], args['img_size']))  # Resize to the target size

    # Transform the image for the model
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU

    with torch.no_grad():
        # Run the model on the input image
        output = net(image)

        # Save the prediction
        save_prediction(output, filename='21_training_pred')
        print("Prediction saved successfully!")


if __name__ == '__main__':
    predict()
    thresh_OTSU(args['pred_path'] + 'pred/')
