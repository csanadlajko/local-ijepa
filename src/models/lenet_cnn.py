import datetime

import matplotlib.pyplot as plt
import torch
from torch import nn

from local_jepa import get_dataset

from ..parser.parser import parse_jepa_args

args = parse_jepa_args()

datasets = get_dataset(args.dataset, args.dataset_input)

if "error" in datasets:
    raise FileNotFoundError(datasets["error"])

train_loader, test_loader = datasets["train_loader"], datasets["test_loader"]

run_identifier: str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

class LeNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnnmodel = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(13456, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, args.num_classes)
        )

    def forward(self, x):
        x = self.cnnmodel(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LeNet().to(device)

criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

def train():
    total, correct = 0, 0
    total_loss = 0
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        y_pred = model.forward(imgs)

        loss = criterion(y_pred, labels)
        loss.backward()
        optim.step()
        optim.zero_grad()

        total_loss += loss.item()

        _, predicted = torch.max(y_pred, 1)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()
        
        if batch_idx % 100 == 0:
            print(f"loss at batch: {batch_idx} - {loss.item()}")

    accuracy = (correct / total) * 100
    epoch_loss = total_loss / total
    return accuracy, epoch_loss


def eval_lenet(train_data, model):
    total, correct = 0, 0
    for batch_idx, (imgs, labels) in enumerate(train_data):
        imgs = imgs.to(device)
        labels = labels.to(device)
        y_pred = model.forward(imgs)
        _, predicted = torch.max(y_pred, 1)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        if batch_idx % 500 == 0:
            print(f"running accuracy: {correct / total * 100}")

    
    return 100 * ( correct / total)

def show_lenet_accuracy(accuracy_list: list):
    epoch_list = range(1, len(accuracy_list) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epoch_list, accuracy_list, label="Accuracy per epoch")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy per epoch (%)')
    plt.title("LeNet accuracy per epoch (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/{args.result_folder}/lenet_accuracy_plot_{run_identifier}.png', dpi=300)
    plt.show()

def show_lenet_loss(loss_list: list):
    epoch_list = range(1, len(loss_list) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epoch_list, loss_list, label="Loss per epoch")
    plt.xlabel('Epochs')
    plt.ylabel('Loss per epoch (Cross Entropy)')
    plt.title("LeNet loss per epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/{args.result_folder}/lenet_loss_plot_{run_identifier}.png', dpi=300)
    plt.show()

acc_list = []
loss_list = []

print(f"starting training LeNet CNN model with: {sum(p.numel() for p in model.parameters())}")

for i in range(args.epochs):
    running_acc, running_loss = train()
    acc_list.append(running_acc)
    loss_list.append(running_loss)
    print(f"epoch {i+1} done")

torch.save(model.state_dict(), f"/{args.result_folder}/trained_lenet_{run_identifier}.pth")

show_lenet_loss(loss_list)
show_lenet_accuracy(acc_list)

acc = eval_lenet(test_loader, model)

print(f"LeNet final accuracy: {acc:.4f}%")