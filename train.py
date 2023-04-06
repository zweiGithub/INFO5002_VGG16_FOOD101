import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import Food101
from PIL import Image
from torchmetrics import Accuracy
from timeit import default_timer as timer
from tqdm.auto import tqdm
from modules import VGG16
from torch.cuda.amp import autocast, GradScaler
#from torch.utils.tensorboard import SummaryWriter
import utils

def main():
  
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Define the necessary data transforms
  train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
  ])

  # Download and load the Food 101 dataset
  food_train = Food101(root='./data', split='train', download=True, transform=train_transforms)
  food_test = Food101(root='./data', split='test', download=True, transform=val_transforms)

  class_names = food_train.classes

  batch_size = 64
  num_workers = os.cpu_count()

  # Create the data loaders
  train_loader = DataLoader(food_train, batch_size=batch_size, shuffle=True,num_workers=num_workers)
  test_loader = DataLoader(food_test, batch_size=batch_size, shuffle=False,num_workers=num_workers)
  # Define accuracy function
  accuracy_fn = Accuracy(task="multiclass", num_classes=len(class_names)).to(device)


  #************STRAT THE TRAINING***************#

  torch.manual_seed(42)
  torch.cuda.manual_seed(42)

  train_time_start = timer()

  #init the model
  current_model = VGG16(num_classes=len(class_names)).to(device)


  epochs =10
  module_dir = ".\models"

  # Define the loss function and optimizer
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.SGD(current_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

  try:
    # Load the latest checkpoint
    latest_checkpoint = utils.latest_checkpoint_path(module_dir,current_model.__class__.__name__ + "_*.pth")
    current_model, optimizer, _, epoch_str = utils.load_model(model=current_model,
                                              optimizer=optimizer,
                                              learning_rate=0.1,
                                              save_path=latest_checkpoint)
    epoch_str = max(1, int(epoch_str))
  except:
    print("No checkpoint found, training from scratch")
    epoch_str = 0
    pass

  # Define a learning rate scheduler
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

  # start training loop
  for epoch in tqdm(range(epoch_str,epochs + 1)):
    train_and_evaluate(model=current_model,
              loaders=[train_loader,test_loader],
              loss_fn=loss_fn,
              optimizer=optimizer,
              accuracy_fn=accuracy_fn,
              device=device,
              epoch=epoch,
              scheduler=scheduler,
              model_dir=module_dir)

    scheduler.step()

  train_time_end= timer()
  print_train_time(train_time_start,train_time_end,device)
  
def train_and_evaluate(model: torch.nn.Module,
               loaders: list,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device,
               epoch: int,
               scheduler: torch.optim.lr_scheduler,
               model_dir: str):
   
    train_loss, train_acc = 0, 0
    train_loader , eval_loader= loaders

    model.train()
    # Create a GradScaler instance for automatic scaling of gradients
    scaler = GradScaler()

    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # autocast context for mixed precision training
        with autocast():
            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss
            loss = loss_fn(y_pred, y)

        train_loss += loss.item()
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward with gradient scaling
        scaler.scale(loss).backward()

        # 5. Optimizer step with gradient scaling
        scaler.step(optimizer)

        # 6. Update the scaler
        scaler.update()

        if batch % 400 == 0:
          print(f"\nLook at batch: {batch}/{len(train_loader)}")

    # Divide total train loss by length of train dataloader
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    print(f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}% | learning rate: {optimizer.param_groups[0]['lr']:.5f}")

    if epoch % 5 == 0:
      eval_summary = eval_model(model=model,
                data_loader=eval_loader,
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
                device=device)
      print(eval_summary)
      utils.save_model(model=model,
                optimizer=optimizer,
                learning_rate=scheduler.get_last_lr()[0],
                iteration=epoch,
                save_path=f"{model_dir}\{model.__class__.__name__}_{epoch}.pth")

def eval_model(model:torch.nn.Module,
               data_loader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               accuracy_fn,
               device):
  loss,acc = 0,0
  model.eval()
  with torch.inference_mode():
    for batch, (X,y) in enumerate(data_loader):
      X,y = X.to(device),y.to(device)
      #make pred
      y_pred = model(X)
      loss += loss_fn(y_pred,y)
      acc += accuracy_fn(y,y_pred.argmax(dim=1))
    #loss per batch
    loss /= len(data_loader)
    acc /= len(data_loader)
  
  return {"model_name":model.__class__.__name__,
          "model_loss":loss.item(),
          "model_acc": acc.item()}

def print_train_time(start:float,
                     end:float,
                     device:torch.device=None):
  total_time=end-start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  return total_time

if __name__ == "__main__":
  main()