import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from modules import VGG16,TinyVGG
import utils
from PIL import Image
import matplotlib.pyplot as plt
import random


def run_inference():
    #Load the class names
    # class_names,class_idx = utils.find_classes("./data/food-101/images")
    #Target image
    # image_path = "./data/food-101/images/apple_pie/134.jpg"
    #Transforms
    # transform = transforms.Compose([
    #     transforms.Resize(size = (224,224)),
    #     transforms.ToTensor(),
    # ])
    #Load the image and convert it to a tensor
    # image = Image.open(image_path)
    # target_tensor = transform(image)


    datasets = torchvision.datasets.FashionMNIST(root="./data", train=False, download=False, transform=transforms.ToTensor())
    class_names = datasets.classes
    #Get a random image
    target_tensor,label = datasets[random.randint(0,len(datasets))]
    #Load the model
    model = TinyVGG(input_shape=1,hidden_units=10,output_shape=len(class_names))
    #Load the latest checkpoint
    model,_,_,_ = utils.load_model(model, None, 0.001, utils.latest_checkpoint_path("./checkpoints", "TinyVGG_*.pth"))
    #Set the model to eval mode
    model.eval()
    with torch.inference_mode():
        #Make a prediction
        prediction_logit = model(target_tensor.unsqueeze(0))
        prediction_probs = torch.softmax(prediction_logit, dim=1)
        prediction_label_idx = torch.argmax(prediction_probs,dim=1)
        #Get the class name
        class_name = class_names[prediction_label_idx]

    plt.figure(figsize=(3,3))
    plt.imshow(target_tensor.squeeze(), cmap="gray")
    plt.title(f"Prediction: {class_name} | Label: {class_names[label]}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    run_inference()


