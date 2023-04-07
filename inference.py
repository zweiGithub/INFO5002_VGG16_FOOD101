import torch
import torchvision
import torchvision.transforms as transforms
from modules import VGG16
import utils
from PIL import Image
import matplotlib.pyplot as plt


def run_inference():
    #Load the class names
    class_names,class_idx = utils.find_classes("./data/food-101/images")
    #Target image
    image_path = "./data/food-101/images/pancakes/7498.jpg"
    #Transforms
    transform = transforms.Compose([
        transforms.Resize(size = (224,224)),
        transforms.ToTensor(),
    ])
    #Load the image and convert it to a tensor
    image = Image.open(image_path)
    traget_tensor = transform(image)
    #Load the model
    model = VGG16(num_classes=101)
    #Load the latest checkpoint
    model,_,_,_ = utils.load_model(model, None, 0.001, utils.latest_checkpoint_path("./checkpoints", "VGG16_*.pth"))
    #Set the model to eval mode
    model.eval()
    with torch.inference_mode():
        #Make a prediction
        prediction_logit = model(traget_tensor.unsqueeze(0))
        prediction_probs = torch.softmax(prediction_logit, dim=1)
        prediction_label_idx = torch.argmax(prediction_probs,dim=1)
        #Get the class name
        class_name = class_names[prediction_label_idx]

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.title(f"Prediction: {class_name}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    run_inference()


