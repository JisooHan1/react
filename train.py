import torch
import torch.nn as nn
import torch.optim as optim
import os
from util.args_loader import get_args
from util.data_loader import transform_train, transform_test
from util.model_loader import get_model
import torchvision
import torchvision.datasets as datasets

def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy

def main():
    # get args
    args = get_args()

    # set GPU
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # set data loader
    train_dataset = datasets.CIFAR10(
        root='./datasets/id_data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(
        root='./datasets/id_data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # set model
    model = get_model(args, num_classes=10, load_ckpt=False)
    model = model.to(device)

    # set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # set checkpoint directory
    checkpoint_dir = f"checkpoints/{args.in_dataset}/{args.name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_acc = 0
    # start training
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch, device)
        accuracy = test(model, test_loader, criterion, device)
        scheduler.step()

        # save checkpoint
        if accuracy > best_acc:
            best_acc = accuracy
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'accuracy': accuracy,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint,
                      f'{checkpoint_dir}/checkpoint_{args.epochs}.pth.tar')
            print(f'Checkpoint saved. Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()