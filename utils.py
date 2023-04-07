import torch
import torchvision
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image


def print_train_time(start:float,
                     end:float,
                     device:torch.device=None):
  total_time=end-start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  return total_time

def save_model(model:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               learning_rate:float,
               iteration:int,
               save_path:str):
  
  state_dict = model.state_dict()
  optimizer_dict = optimizer.state_dict() if optimizer is not None else None

  torch.save({'model_state_dict':state_dict,
              'iteration':iteration,
              'optimizer':optimizer_dict,
              'learning_rate':learning_rate},
              save_path)
  
  print(f"Model saved to {save_path}")

def load_model(model:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               learning_rate:float,
               save_path:str):
  
  assert os.path.isfile(save_path)

  checkpoint_dict = torch.load(save_path,map_location="cpu")

  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']

  if optimizer is not None and checkpoint_dict['optimizer'] is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])

  model.load_state_dict(checkpoint_dict['model_state_dict'])

  print(f"Model loaded from {save_path}")
  return model,optimizer,learning_rate,iteration

# Define a function that returns the latest checkpoint in a directory
def latest_checkpoint_path(dir_path, regex):
  # Get a list of all checkpoint files in the directory
  f_list = glob.glob(os.path.join(dir_path, regex))
  # Sort the list of checkpoints by the number in the file name
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  # Select the last checkpoint in the list
  x = f_list[-1]
  # Print the path of the checkpoint
  print(x)
  # Return the path of the checkpoint
  return x

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def find_classes(dir):
  classes = sorted(entry.name for entry in os.scandir(dir) if entry.is_dir())
  classes_idx = {classes[i]: i for i in range(len(classes))}
  return classes, classes_idx

def display_random_image(
        model: torch.nn.Module,
        dataset:torchvision.datasets,
        class_names:list,
        device:torch.device,
        num_images:int=9,
        seed:int=42):
  if num_images > 9:
    num_images=9
  if seed:
    random.seed(seed)
  random_samples_idx = random.sample(range(len(dataset)), num_images)
  fig = plt.figure(figsize=(9, 9))
  plt.axis("off")
  nrows = 3
  ncols = 3
  i = 0
  for idx in random_samples_idx:
    img, label = dataset[idx]
    img = img.unsqueeze(0)
    img = img.to(device)
    model.eval()
    with torch.no_grad():
      pred_logit = model(img)
      pred_probs = torch.softmax(pred_logit, dim=1)
      pred_label_idx = torch.argmax(pred_probs, dim=1)
      pred_label = class_names[pred_label_idx.item()]
      true_label = class_names[label]
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0))
    plt.title(f"True label: {true_label} \n Predicted label: {pred_label} \n Predicted Label Probs: {torch.max(pred_probs).item():.2f}")
    i += 1
  return fig

