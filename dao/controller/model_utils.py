import dao.controller.models as models
import torch
from dao.controller.image_utils import preprocess_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(model_path, input_channels):
    """
    Loads a PyTorch model onto the GPU.
    """
    model = models.UNet11(input_channels=input_channels)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model


def run_model(patch, model):
    """
    Executes a PyTorch model.
    """
    model.eval()
    # print("Model in eval mode")
    with torch.set_grad_enabled(False):
        response = torch.sigmoid(model(patch))
    return response



def image_loader(img, satelite=2):
    """
    Preprocesses an image to make it suitable for input into a segmentation model.
    """
    img = preprocess_image(img, satelite)
    return img


def predict(model, image):
    """
    Processes a preprocessed satellite image using a segmentation model to
    generate a segmentation mask.
    """

    img_input = image_loader(image, 2)
    trained_model = model
    mask_predicted = run_model(img_input, trained_model)
    del trained_model

    return mask_predicted.cpu().numpy()
