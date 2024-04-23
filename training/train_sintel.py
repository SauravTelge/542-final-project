import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from torch.utils.data import DataLoader
# from loss import get_flow_loss
from tqdm import trange, tqdm
from Sintel_dataset import make_dataset, ListDataset
from loss import realEPE, mse_loss
import argparse
from torch.utils.tensorboard import SummaryWriter
import os

writer = SummaryWriter('tensor_logs_new')
# writer = SummaryWriter('log_dir')

parser = argparse.ArgumentParser(
        description='''Train U-Net for Optical Flow.''')
    
parser.add_argument('--data', type=str, default='data',
                    help='''path to the training data folder''')
parser.add_argument('--save_path', default='checkpoints', type=str, 
                    help='path to save checkpoints')
parser.add_argument('--epochs', default=100, type=int, 
                    help='number of epochs')
parser.add_argument('--batch_size', default=2, type=int, 
                    help='batch size')
parser.add_argument('--checkpoint_path', default='', 
                    help='path of saved checkpoint')
parser.add_argument('--prev_step', default=0, type=int,
                    help='previous step of tensorlog file')
args = parser.parse_args()

#args.data = "/home/rutwik/542project/dift/project_sintel_files/Sintel"

X_train, X_test, y_train, y_test = make_dataset(args.data,dataset_type="clean")
train_dataset = ListDataset(args.data, X_train, y_train)
test_dataset = ListDataset(args.data, X_test, y_test)
checkpoint_path = args.save_path

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
print("Data loaded")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(torch.cuda.get_device_name())
# print(torch.cuda.current_device())
# device_id = torch.cuda.current_device()
num_epochs = args.epochs

# model = Model()
model = Model().cuda()
# model = model.to(device=device)
optimizer = optim.Adam(list(model.parameters()), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
start_epoch = 0
if args.checkpoint_path != '':
    print('Loading checkpoint')
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    # model = model.load_state_dict(torch.load(args.checkpoint_path, map_location='cuda'))
    # optimizer = optimizer.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))
    # checkpoint_cuda = {}
    # for key, value in checkpoint.items():
    #     if key != 'model_state_dict': continue
    #     if isinstance(value, torch.Tensor):
    #         checkpoint_cuda[key] = value.to('cuda')
    #     else:
    #         checkpoint_cuda[key] = value
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # model.load_state_dict(checkpoint_cuda['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to('cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    running_loss = checkpoint['loss']
    assert start_epoch < num_epochs
    assert start_epoch <= args.prev_step
    start_epoch = max(start_epoch,args.prev_step)

min_val_loss = torch.inf
min_loss = torch.inf
for epoch in range(start_epoch, num_epochs):
    model.train()
    # print(f"epoch started: {epoch+1}")
    
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    if epoch==0:
        prev_loss = 0
    else:
        prev_loss = running_loss
    running_loss = 0.0
    for data in pbar:
       
        inputs, targets = data['features'], data['flow'].cuda()

        for i in range(len(inputs)):
            inputs[i] = inputs[i].cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(inputs)
        # outputs = torch.transpose(outputs, 3, 2)
        # print(f"output: {outputs.shape}")
        # print(f"target: {targets.shape}")
        # loss = realEPE(outputs,targets)
        loss = mse_loss(outputs,targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()/len(pbar)
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        pbar.set_postfix(Loss= loss.item(), Prev_epoch_loss=prev_loss)

    writer.add_scalar('training loss', running_loss, epoch+1)
        
        # write validation code here that runs after every 5 epochs
    
    if (((epoch + 1) % 2)!=0): continue
    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        running_loss_val = 0
        for data in pbar:
       
            inputs, targets = data['features'], data['flow'].cuda()

            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda()
            
            # forward + backward + optimize
            outputs = model(inputs)
            # outputs = torch.transpose(outputs, 3, 2)
            # loss_val = realEPE(outputs,targets)
            loss_val = mse_loss(outputs,targets)

            running_loss_val += loss_val.item()/len(pbar)
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
            pbar.set_postfix(Loss= loss_val.item())

    writer.add_scalar('validation loss', running_loss_val, epoch+1)
    # print(f"Training loss after epoch {epoch}: {running_loss}")
    # print(f"Validation loss after epoch {epoch}: {running_loss_val}")

    # if (((epoch + 1) % 5)==0):
    if (running_loss_val < min_val_loss):
    # if (running_loss < min_loss):
        min_val_loss = running_loss_val
        # min_loss = running_loss

        if len(os.listdir(checkpoint_path)) != 0: 
            # print("deleting previous check point, sed")
            previous_file = os.listdir(checkpoint_path)[0]
            os.remove(os.path.join(checkpoint_path, previous_file))
        # print(f"Saving checkpoint at")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            }, os.path.join(checkpoint_path, f'ckpt_{epoch + 1}.pt'))
        check_path = os.path.join(checkpoint_path, f'ckpt_{epoch + 1}.pt')
        # print(f'Checkpoint saved at: {check_path}')

print('Finished Training')