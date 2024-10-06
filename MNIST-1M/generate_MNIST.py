from tqdm import tqdm
import os, sys
import torch
import numpy as np
import argparse 

from MNIST_script import ContextUnet, DDPM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MNIST images using DDPM.')
    parser.add_argument('--dataset_size', type=int, default=1_000_000, help='Size of the dataset to generate.')
    parser.add_argument('--n_T', type=int, default=400, help='Number of time steps in DDPM.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use for training.')
    parser.add_argument('--n_classes', type=int, default=10, help='Number of classes in the dataset.')
    parser.add_argument('--n_feat', type=int, default=128, help='Number of features in the model.')
    parser.add_argument('--model_path', type=str, default='model_39.pth', help='Path to the trained model.')
    parser.add_argument('--savedir', type=str, default='/n/holyscratch01/pehlevan_lab/Lab/aatanasov/MNIST_1M/', help='Directory to save the generated images.')
    parser.add_argument('--w', type=float, default=0.5, help='Weight of the guide loss.')
    args = parser.parse_args()

    total_dataset_size = args.dataset_size
    n_T = args.n_T
    device = args.device
    n_classes = args.n_classes
    n_feat = args.n_feat
    model_path = args.model_path
    savedir = args.savedir
    w = args.w

    model = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    model.load_state_dict(torch.load(model_path)) 
    model.eval()
 
    chunk_size = 100_000 
    num_chunks = total_dataset_size // chunk_size
    model_batch = 1000
    with torch.no_grad():
        for chunk in range(num_chunks):
            print(f"Chunk number: {chunk} of {num_chunks}")
            x_gen_total = []
            y_gen_total = []
            for _ in tqdm(range(chunk_size//model_batch)):
                x_gen, _ = model.sample(model_batch, (1, 28, 28), device, guide_w=w)
                y_gen = np.tile(np.arange(10), chunk_size // 10)
                perm = torch.randperm(x_gen.size(0))
                x_gen, y_gen = x_gen[perm], y_gen[perm]
                x_gen_total.append(x_gen.cpu().detach().numpy())
                y_gen_total.append(y_gen)
            
            sys.stdout.flush()

            x_gen_total = np.concatenate(x_gen_total, axis=0)
            y_gen_total = np.concatenate(y_gen_total, axis=0)
            np.savez(savedir+f'MNIST_images_{chunk}.npz', x_gen_total)
            np.save(savedir+f'MNIST_labels_{chunk}.npz', y_gen_total)

        print("Image generation complete.")