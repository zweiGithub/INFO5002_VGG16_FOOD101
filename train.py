import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import Food101, FashionMNIST
from PIL import Image
from torchmetrics import Accuracy
from timeit import default_timer as timer
from tqdm.auto import tqdm
from modules import VGG16,TinyVGG
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import utils

global_step = 0
INTERVAL_STEPS = 5000
BATCH_SIZE = 64

def main():
  
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Define the necessary data transforms
  train_transforms = transforms.Compose([
    transforms.Resize(size = (224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  val_transforms = transforms.Compose([
    transforms.Resize(size = (224,224)),
    transforms.ToTensor(),
  ])

  current_pwd = os.getcwd()

  # Download and load the Food 101 dataset
  # food_train = Food101(root=f'{current_pwd}/data', split='train', download=True, transform=train_transforms)
  # food_test = Food101(root=f'{current_pwd}/data', split='test', download=True, transform=val_transforms)
  fashion_train = FashionMNIST(root=f'{current_pwd}/data', train=True, download=True, transform=ToTensor())
  fashion_test = FashionMNIST(root=f'{current_pwd}/data', train=False, download=True, transform=ToTensor())

  # class_names,class_idx = utils.find_classes(f"{current_pwd}/data/food-101/images")
  class_names = fashion_train.classes
  # print(f"Class names: {len(class_names)}")

  num_workers = os.cpu_count()

  # Create the data loaders
  train_loader = DataLoader(fashion_train,batch_size=BATCH_SIZE, shuffle=True,num_workers=num_workers)
  test_loader = DataLoader(fashion_test, batch_size=BATCH_SIZE, shuffle=False,num_workers=num_workers)
  # Define accuracy function
  # accuracy_fn = Accuracy(task="multiclass", num_classes=len(class_names)).to(device)
  accuracy_fn = utils.accuracy_fn


  #************STRAT THE TRAINING***************#

  torch.manual_seed(42)
  torch.cuda.manual_seed(42)

  train_time_start = timer()
  global global_step
  #init the model
  # current_model = VGG16(num_classes=len(class_names)).to(device)
  # current_model = models.resnet50(num_classes=len(class_names)).to(device)
  current_model = TinyVGG(input_shape=1,hidden_units=10,output_shape=len(class_names)).to(device)
  

  writer = SummaryWriter(log_dir=f"{current_pwd}/logs")
  writer_eval = SummaryWriter(log_dir=f"{current_pwd}/logs/eval")

  epochs =50
  module_dir = f"{current_pwd}/checkpoints"

  # Define the loss function and optimizer
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.SGD(current_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
  # optimizer = optim.Adam(current_model.parameters(), lr=0.0001)

  try:
    # Load the latest checkpoint
    latest_checkpoint = utils.latest_checkpoint_path(module_dir,current_model.__class__.__name__ + "_*.pth")
    current_model, optimizer, _, epoch_str = utils.load_model(model=current_model,
                                              optimizer=optimizer,
                                              learning_rate=0.1,
                                              save_path=latest_checkpoint)
    epoch_str = max(1, int(epoch_str))
    global_step = (epoch_str-1) * len(train_loader)
  except:
    print("No checkpoint found, training from scratch")
    epoch_str = 0
    global_step = 0
    pass

  # Define a learning rate scheduler
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
  # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1, last_epoch=epoch_str - 1)

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
              model_dir=module_dir,
              writers=[writer,writer_eval],
              datasets=fashion_test,
              class_names=class_names)

    scheduler.step()

  train_time_end= timer()
  utils.print_train_time(train_time_start,train_time_end,device)
  
def train_and_evaluate(model: torch.nn.Module,
               loaders: list,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device,
               epoch: int,
               scheduler: torch.optim.lr_scheduler,
               model_dir: str,
               writers: list,
               datasets:dict,
               class_names: list):
   
    train_loss, train_acc = 0, 0
    train_loader , eval_loader= loaders
    writer, writer_eval = writers
    global global_step

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
            acc = accuracy_fn(y, y_pred.argmax(dim=1))

        train_loss += loss.item()
        train_acc += acc

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward with gradient scaling
        scaler.scale(loss).backward()

        # 5. Optimizer step with gradient scaling
        scaler.step(optimizer)

        # 6. Update the scaler
        scaler.update()

        if global_step % INTERVAL_STEPS == 0 and global_step != 0:
          # 7. add loss and accuracy to tensorboard
          print(f"\nTrain epoch :{epoch} [{100.*batch/len(train_loader):.0f}%]")
          eval_model(model=model,
                data_loader=eval_loader,
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
                device=device,
                writer=writer_eval,
                datasets=datasets,
                class_names=class_names,)

          utils.save_model(model=model,
                optimizer=optimizer,
                learning_rate=scheduler.get_last_lr()[0],
                iteration=epoch,
                save_path=f"{model_dir}/{model.__class__.__name__}_{global_step}.pth")
        global_step += 1

    # Divide total train loss by length of train dataloader
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    writer.add_scalar("Loss/train", train_loss, global_step)
    writer.add_scalar("Accuracy/train(%)", train_acc, global_step)
    writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], global_step)
    print(f"\nTrain Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}% | learning rate: {optimizer.param_groups[0]['lr']:.5f}")


def eval_model(model:torch.nn.Module,
               data_loader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               accuracy_fn,
               device,
               writer:SummaryWriter,
               datasets:dict,
               class_names:list):
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
  
  writer.add_scalar("Loss/eval", loss, global_step)
  writer.add_scalar("Accuracy/eval(%)", acc, global_step)
  writer.add_figure("Predictions vs. actuals",figure = utils.display_random_image(model=model,dataset=datasets,class_names=class_names,device=device),global_step=global_step)


if __name__ == "__main__":
  main()