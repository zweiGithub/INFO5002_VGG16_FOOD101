import torch
import glob
import os

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