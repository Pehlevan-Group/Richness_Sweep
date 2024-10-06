import os, sys, argparse, gc, time, pickle, glob, pytz, gc
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ConstantLR

from collections import defaultdict

from models import MLP, CNN, ResNet18, CenteredModel
from data_loaders import load_file_MNIST_1M, load_file_CIFAR_5M

from utils import default_factory, generate_logspace

# Set CUDA allocation configuration to manage fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

proj_dir = "/n/holyscratch01/pehlevan_lab/Everyone/gamma_sweep/"
models_dir = "/n/holyscratch01/pehlevan_lab/Everyone/gamma_sweep/model_files/"
loss_dir = "/n/holyscratch01/pehlevan_lab/Everyone/gamma_sweep/loss_dicts/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)

def time_now():
  return datetime.now(pytz.timezone('US/Eastern')).strftime("%m-%d_%H-%M")

def time_diff(t_a, t_b):
  t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
  return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

def test_model(model, test_data, test_labels, loss='mse', test_batch_size=1000, doubling=False):
    model.eval()
    test_loss = 0
    correct = 0
    if loss=='xent':
        test_criterion = nn.CrossEntropyLoss(reduction='sum')
    elif loss=='mse':
        test_criterion = nn.MSELoss(reduction='sum')
    else:
        raise ValueError('Invalid loss function')
    with torch.no_grad():
        for i in range(len(test_data)//test_batch_size):
            data = test_data[test_batch_size*i:test_batch_size*(i+1)]
            target = test_labels[test_batch_size*i:test_batch_size*(i+1)]
            if doubling: 
                data, target = data.double(), target.double()
            true_label = torch.argmax(target, dim=1)
            if loss=='xent':
                target = true_label
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_loss += test_criterion(output, target).item()
            correct += pred.eq(true_label.view_as(pred)).sum().item()
    test_loss /= len(test_data)
    correct /= len(test_data)
    model.train()
    return test_loss, correct

def save_model_dict(model, model_dir, file_name):
    os.makedirs(model_dir, exist_ok=True)
    file_path = os.path.join(model_dir, file_name)
    torch.save(model.state_dict(), file_path)

def get_train_data(dataset, file_num):
    if dataset == 'MNIST-1M':
        X_train, y_train = load_file_MNIST_1M(file_num=file_num)
    elif dataset == 'CIFAR-5M':
        X_train, y_train = load_file_CIFAR_5M(file_num=file_num)
    return X_train, y_train

def get_test_data(dataset, P_test):
    if dataset == 'MNIST-1M':
        test_data = torch.load(f'MNIST/test_data.pt')
        test_labels = torch.load(f'MNIST/test_labels.pt')
        test_labels = torch.Tensor(test_labels).long()
        test_labels = F.one_hot(test_labels, num_classes=10).float()
    elif dataset == 'CIFAR-5M':
        test_data = torch.load(f'CIFAR/test_data.pt')
        test_labels = torch.load(f'CIFAR/test_labels.pt')
        test_labels = torch.Tensor(test_labels).long()
        test_labels = F.one_hot(test_labels, num_classes=10).float()
    return test_data[:P_test], test_labels[:P_test]

def process_chunk(
    chunk_num,
    model_dir, arch, model_kwargs,
    opt_class, 
    dataset, device, N, B, 
    loss_type, 
    E, d, 
    test_data, test_labels,
    gamma0s, eta0s, 
    eta0_range, range_below_max_eta0, 
    loss_dict, dict_file, start_time, 
    save_model,
):
    lspace = generate_logspace(25)

    # Load new chunk data
    train_data, train_labels = get_train_data(dataset, chunk_num)
    train_data, train_labels = train_data.to(device), train_labels.to(device)
    
    for gamma0 in gamma0s:
        if gamma0 < 1e-2: 
            doubling = True
        else:
            doubling = False
        stable_lr = False
        terminal_log_eta0 = -eta0_range
        for eta0 in eta0s:
            if eta0 < np.power(10.0, float(terminal_log_eta0)):
                break
            loss_nan = False
            for e in range(E):
                torch.manual_seed(e)
                
                model = CenteredModel(arch, gamma0=gamma0, **model_kwargs).to(device)
                if doubling:
                    model = model.double()
                if chunk_num > 0:
                    last_saved = f'{model_dir}/gamma0={gamma0:.0e}_eta0={eta0:.0e}_e={e}_d={d}_t=-1.pth'
                    if os.path.exists(last_saved):
                        model.load_state_dict(torch.load(last_saved))
                    else:
                        loss_nan = True
                        break
                optimizer = opt_class(model.parameters(), lr= N * eta0)
                model.train()

                for t in range(len(train_data) // B):
                    t_eff = t + chunk_num * len(train_data) // B
                    data, target = train_data[B*t:B*(t+1)], train_labels[B*t:B*(t+1)]
                    if doubling:
                        data, target = data.double(), target.double()
                    
                    if t_eff in lspace and save_model:
                        test_loss, test_acc = test_model(model, test_data, test_labels, loss=loss_type, doubling=doubling)
                        loss_dict[gamma0, eta0][f'test_loss_{e}'].append(test_loss)
                        loss_dict[gamma0, eta0][f'test_accuracy_{e}'].append(test_acc)
                        file_name = f'gamma0={gamma0:.0e}_eta0={eta0:.0e}_e={e}_d={d}_t={t_eff}.pth'
                        save_model_dict(model, model_dir, file_name)
                    
                    if loss_type == 'xent':
                        target = target.argmax(dim=1)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    loss_dict[gamma0, eta0][f'train_loss_{e}'].append(loss.item())
                    loss_nan = torch.isnan(loss).item()
                    if loss_nan:
                        break
                
                if save_model:
                    test_loss, test_acc = test_model(model, test_data, test_labels, loss=loss_type, doubling=doubling)
                    loss_dict[gamma0, eta0][f'test_loss_{e}'].append(test_loss)
                    loss_dict[gamma0, eta0][f'test_accuracy_{e}'].append(test_acc)
                    file_name = f'gamma0={gamma0:.0e}_eta0={eta0:.0e}_e={e}_d={d}_t=-1.pth'
                    save_model_dict(model, model_dir, file_name)
                
                if loss_nan:
                    # Clean up loss_dict and saved models
                    if chunk_num == 0:
                        to_del = [k for k in list(loss_dict.keys()) if k[0] == gamma0 and k[1] >= eta0]
                    else:
                        to_del = [k for k in list(loss_dict.keys()) if k[0] == gamma0 and k[1] == eta0]
                    for k in to_del:
                        del loss_dict[k]
                    if save_model:
                        for k in to_del:
                            pattern = os.path.join(model_dir, f'gamma0={k[0]:.0e}_eta0={k[1]:.0e}_e={e}_d={d}_t=*')
                            for f in glob.glob(pattern):
                                os.remove(f)
                    stable_lr = False
                    terminal_log_eta0 = -eta0_range
                    break
                
                # Memory Management Steps
                model.cpu()                # Move model to CPU
                del model                  # Delete the model instance
                del optimizer              # Delete the optimizer instance
                model = None               # Remove reference
                optimizer = None           # Remove reference
            torch.cuda.empty_cache()       # Clear cached GPU memory
            gc.collect()                   # Trigger garbage collection
                
            if not stable_lr and (not loss_nan):
                stable_lr = True
                if loss_type == 'xent' and gamma0 < 1e-0:
                    terminal_log_eta0 = -eta0_range
                else:
                    terminal_log_eta0 = max(np.log10(eta0) - range_below_max_eta0, -eta0_range)
                
            time_elapsed = time.time() - start_time
            if not loss_nan:
                print(f"eta0 = {eta0:.0e}, gamma0 = {gamma0:.0e} | Time elapsed: {time.strftime('%H:%M:%S', time.gmtime(time_elapsed))}")
            sys.stdout.flush()
        
        with open(dict_file + '.pkl', 'wb') as f:
            pickle.dump(loss_dict, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='All parameter for the gamma sweep')
    
    # Architecture
    parser.add_argument('--arch', type=str, required=True, default='MLP', help="Architecture")
    parser.add_argument('-N', type=int, required=True, default=1000, help="Width")
    parser.add_argument('-L', type=int, required=False, default=3, help="Depth")
    parser.add_argument('-E', type=int, required=True, default=1, help="Num Ensembles")
    # Data
    parser.add_argument('--task', type=str, required=True, default='MNIST-1M', help="Task")
    parser.add_argument('-s', type=float, required=False, default=0, help="Noise level")
    parser.add_argument('-d', type=int, required=False, default=0, help="Data key")
    # Optimization
    parser.add_argument('--optimizer', type=str, required=False, default='sgd', help="Optimizer")
    parser.add_argument('--loss', type=str, required=False, default='mse')
    parser.add_argument('-B', type=int, required=True, default=1, help="Batch Size")
    parser.add_argument('--eta0_range', type=int, required=False, default=12, help="Log10 range of eta0")
    parser.add_argument('--gamma0_range', type=int, required=False, default=5, help="Log10 range of gamma0")
    parser.add_argument('--eta0_res', type=int, required=False, default=2, help="Resolution of eta0")
    parser.add_argument('--gamma0_res', type=int, required=False, default=2, help="Resolution of gamma0")
    parser.add_argument('--range_below_max_eta0', type=int, required=False, default=3, help="Log10 range of eta below the max value")
    parser.add_argument('--save_model', action='store_true', help="Save model")

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Model:
    arch_name, N, L = args.arch, args.N, args.L
    if arch_name == 'CNN':
        L = 4
    if arch_name == 'ResNet18':
        L = 3
    dataset = args.task
    B, E = args.B, args.E
    sigma_epsilon = args.s
    d = args.d
    save_model = args.save_model

    # Optimizer
    loss_type = args.loss
    gamma0_range, gamma0_res = args.gamma0_range, args.gamma0_res
    eta0_range, eta0_res = args.eta0_range, args.eta0_res
    gamma0s = np.logspace(-gamma0_range, gamma0_range, 1 + 2 * gamma0_range * gamma0_res)
    eta0s = np.logspace(eta0_range, -eta0_range, 1 + 2 * eta0_range * eta0_res, -1)
    range_below_max_eta0 = args.range_below_max_eta0
    opt_string = args.optimizer
    opt_class = optim.SGD

    prefix = f'{arch_name}_{dataset}_{opt_string}_s={sigma_epsilon:.0e}_N={N}_L={L}_B={B}_loss={loss_type}' 
    model_dir = os.path.join(models_dir, prefix)
    dict_file = os.path.join(loss_dir, f'{prefix}_E={E}_d={d}')
    
    W, H, C = (28, 28, 1) if dataset == 'MNIST-1M' else (32, 32, 3)
    if arch_name == 'MLP':
        model_kwargs = {'D': H * W * C, 'width': N, 'depth': L}
    elif arch_name == 'CNN':
        model_kwargs = {'W': W, 'H': H, 'C': C, 'width': N}
    elif arch_name == 'ResNet18':
        model_kwargs = {'wm': N/64}
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
    arch_dict = {'MLP': MLP, 'CNN': CNN, 'ResNet18': ResNet18 }
    arch = arch_dict[arch_name]

    # Data:
    P_test = 5000
    test_data, test_labels = get_test_data(dataset, P_test)
    test_data, test_labels = test_data.to(device), test_labels.to(device)

    loss_dict = defaultdict(default_factory)
    print(f"B = {B}: ")
    start_time = time.time()
    

    memory_fit = {'MNIST-1M': 1, 'CIFAR-5M': 6}[dataset]
    criterion = {'mse': nn.MSELoss(), 'xent': nn.CrossEntropyLoss()}[loss_type]
    
    for chunk_num in range(memory_fit):
        print("-" * 50)
        print(f"Chunk {chunk_num + 1} out of {memory_fit}")
        process_chunk(
            chunk_num,
            model_dir, arch, model_kwargs,
            opt_class, 
            dataset, device, N, B, 
            loss_type, 
            E, d, 
            test_data, test_labels,
            gamma0s, eta0s,
            eta0_range, range_below_max_eta0, 
            loss_dict, dict_file, start_time, 
            save_model,
        )
        gc.collect()
        torch.cuda.empty_cache()
    
        with open(dict_file + '.pkl', 'wb') as f:
            pickle.dump(loss_dict, f)
    