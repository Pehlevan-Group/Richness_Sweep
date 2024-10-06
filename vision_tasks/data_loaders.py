import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def load_file_CIFAR_5M(savedir="/n/holyscratch01/pehlevan_lab/Everyone/cifar-5m-centered/", file_num=0):
    CIFAR_MEAN = torch.tensor([121.4, 119.3, 109.83]).view(1, 3, 1, 1).to('cuda')
    CIFAR_STD = torch.tensor([63.1, 62.1, 66.6]).view(1, 3, 1, 1).to('cuda')
    file_name = f"{savedir}/cifar5m_part{file_num}.npz"
    print(f"Reading file {file_name}")
    curr_data = np.load(file_name)
    X, y = [curr_data[k] for k in curr_data.keys()]
    train_data = torch.Tensor(X.transpose(0,3,1,2)).to('cuda') 
    train_labels = torch.Tensor(y) 
    with torch.no_grad():
        train_data = (train_data - CIFAR_MEAN) / CIFAR_STD
        
    train_labels = F.one_hot(train_labels.long(), num_classes=10).float()
    return train_data, train_labels

def load_file_MNIST_1M(savedir='/n/holyscratch01/pehlevan_lab/Everyone/MNIST_1M/', file_num=0):
    MEAN = .1307
    STDEV = 0.3081
    images = []
    labels = []
    for i in tqdm(range(10), desc='Loading MNIST-1M data'):
        img_data = np.load(savedir+f'MNIST_images_{i}.npz')['arr_0']
        images.append(img_data )
        label_data = np.load(savedir+f'/MNIST_labels_{i}.npy').astype(int)
        labels.append(label_data)

    train_data = np.concatenate(images, axis=0)
    train_data = (train_data - MEAN)/STDEV
    train_labels = np.concatenate(labels)
    train_data = torch.Tensor(train_data)
    train_labels = torch.Tensor(train_labels).long()
    train_labels = F.one_hot(train_labels, num_classes=10).float()
    return train_data, train_labels

def load_MNIST_1M(savedir='/n/holyscratch01/pehlevan_lab/Everyone/MNIST_1M/', num_files=1, device='cuda'):
    MEAN = .1307
    STDEV = 0.3081
    images = []
    labels = []
    for i in range(num_files):
        img_data = np.load(savedir+f'MNIST_images_{i}.npz')['arr_0']
        images.append(img_data )

        label_data = np.load(savedir+f'/MNIST_labels_{i}.npy').astype(int)
        labels.append(label_data)

    train_data = np.concatenate(images, axis=0)
    train_data = (train_data - MEAN)/STDEV
    train_labels = np.concatenate(labels)
    train_data = torch.Tensor(train_data)
    train_labels = torch.Tensor(train_labels).long()
    train_labels = F.one_hot(train_labels, num_classes=10).float()
    
    test_data = torch.load(f'MNIST/test_data.pt', map_location=device)
    test_labels = torch.load(f'MNIST/test_labels.pt', map_location=device)
    test_labels = F.one_hot(test_labels, num_classes=10).float()
    train_set = {'data': train_data, 'labels': train_labels}
    test_set = {'data': test_data, 'labels': test_labels}
    return train_set, test_set

def load_CIFAR_5M(savedir="/n/holyscratch01/pehlevan_lab/Everyone/cifar-5m-new/", num_files=1, device='cuda'):
    
    CIFAR_MEAN = torch.tensor([125.30691805, 122.95039414, 113.86538318]).view(1, 3, 1, 1).to(device)
    CIFAR_STD = torch.tensor([62.99321928, 62.08870764, 66.70489964]).view(1, 3, 1, 1).to(device)

    print(f"reading cifar 5m data from {savedir}")
    total_size = 0
    for i in range(num_files):
        file_name = f"{savedir}/cifar5m_part{i}.npz"
        curr_data = np.load(file_name)
        X, y = [curr_data[k] for k in curr_data.keys()]
        total_size += y.shape[0]
    
    train_size = int(total_size)
    train_data = np.zeros((train_size, 32, 32, 3), dtype=np.uint8)
    train_labels = np.zeros((train_size,), dtype=int)
    idx = 0 
    for i in range(num_files):
        print("reading file %d" % i)
        file_name = f"{savedir}/cifar5m_part{i}.npz"
        curr_data = np.load(file_name)
        X, y = [curr_data[k] for k in curr_data.keys()]
        len_i = y.shape[0]
        if idx+len_i <= train_size:
            train_data[idx:idx+len_i, ...] = X[:len_i, ...]
            train_labels[idx:idx+len_i] = y[:len_i]
        else:
            break
        idx += len_i

    train_data = torch.Tensor(train_data.transpose(0, 3, 1, 2))
    train_labels = torch.Tensor(train_labels)

    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    train_labels = F.one_hot(train_labels.long(), num_classes=10).float()

    # normalize the PyTorch tensors in each channel by CIFAR_MEAN and CIFAR_STD
    with torch.no_grad():
        train_data = (train_data - CIFAR_MEAN) / CIFAR_STD

    # Load the original CIFAR test set to test against
    test_data = torch.load(f'CIFAR/test_data.pt', map_location=device)
    test_labels = torch.load(f'CIFAR/train_labels.pt', map_location=device)
    test_labels = F.one_hot(test_labels, num_classes=10).float()

    train_set = {'data': train_data, 'labels': train_labels}
    test_set = {'data': test_data, 'labels': test_labels}
    return train_set, test_set

