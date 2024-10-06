import argparse
import os

parser = argparse.ArgumentParser(description='Feed in architecture, task, N, etc')

# Architecture
parser.add_argument('--arch', type=str, required=True, default='MLP', help="Architecture")
parser.add_argument('-N', type=int, required=True, default=1000, help="Width")
parser.add_argument('-L', type=int, required=True, default=3, help="Depth")
parser.add_argument('-E', type=int, required=True, default=1, help="Num Ensembles")
# Data
parser.add_argument('--task', type=str, required=True, default='MNIST-1M', help="Task")
parser.add_argument('-s', type=float, required=False, default=0, help="Noise level")
parser.add_argument('-d', type=int, required=False, default=0, help="Data key")
# Optimization
parser.add_argument('--loss', type=str, required=True, default='mse')
parser.add_argument('-B', type=int, required=True, default=64, help="Batch Size")
parser.add_argument('--optimizer', type=str, required=False, default='sgd', help="Optimizer")
parser.add_argument('--eta0_range', type=int, required=False, default=12, help="Log10 range of eta0")
parser.add_argument('--gamma0_range', type=int, required=False, default=5, help="Log10 range of gamma0")
parser.add_argument('--eta0_res', type=int, required=False, default=2, help="Resolution of eta0")
parser.add_argument('--gamma0_res', type=int, required=False, default=2, help="Resolution of gamma0")

# Saving parameters
parser.add_argument('--save_model', action='store_true', help="Save model")
parser.add_argument('--cpu', action='store_true', help="Whether to use a GPU or not")
parser.add_argument('--kempner', action='store_true', help="Whether to kempner or not")
parser.add_argument('--h100', action='store_true', help="Whether to h100 or not")
args = parser.parse_args()

python_loc = "/n/home00/aatanasov/.conda/envs/scaling4/bin/python"

arch_name, N, L, B = args.arch, args.N, args.L, args.B
dataset = args.task
B, E = args.B, args.E
sigma_epsilon = args.s
loss = args.loss
d = args.d
save_model = args.save_model

print(args)

# Get the directory to which this python script is being run
dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
    
save_str = f'{arch_name}_{dataset}_s={sigma_epsilon:.0e}_N={N}_L={L}_B={B}_loss={loss}_E={E}_d={d}'

# Take the parser variables and convert them into a string of flags to pass to the bash script
flags = ""
for arg in vars(args):
    if arg in 'N L B E s d'.split():
        flags += f"-{arg} {getattr(args, arg)} "
    elif arg in "save_model":
        if getattr(args, arg):
            flags += f"--{arg} "
    elif arg in "arch task optimizer loss eta0_range gamma0_range eta0_res gamma0_res".split():
        flags += f"--{arg} {getattr(args, arg)} "

bash_file_name = dir_path+f"bash_scripts/"+save_str+".sh"

with open (bash_file_name, 'w') as rsh:
    rsh.write('''\
#!/bin/bash
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
''')
    
    if args.task == 'CIFAR-5M' or args.arch == 'ResNet18':
        rsh.write(f"#SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes\n")
    else:
        rsh.write(f"#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes\n")
    if args.cpu:
        rsh.write(f"#SBATCH -p shared, sapphire, pehlevan\n")
        rsh.write(f"#SBATCH --account=pehlevan_gpu\n")
        rsh.write(f"#SBATCH --mem=40G\n")
    else:
        if arch_name == 'ResNet18' or args.h100:
            rsh.write(f"#SBATCH -p kempner_h100\n")
            rsh.write(f"#SBATCH --account=kempner_pehlevan_lab\n")
            rsh.write(f"#SBATCH --gres=gpu:1\n")
            rsh.write(f"#SBATCH --mem-per-gpu=80G\n")
        elif args.kempner: 
            rsh.write(f"#SBATCH -p kempner,kempner_requeue,kempner_h100\n")
            rsh.write(f"#SBATCH --account=kempner_pehlevan_lab\n")
            rsh.write(f"#SBATCH --gres=gpu:1\n")
            rsh.write(f"#SBATCH --mem-per-gpu=40G\n")
        else:
            rsh.write(f"#SBATCH -p gpu,seas_gpu,gpu_requeue\n")
            rsh.write(f"#SBATCH --gres=gpu:1\n")
            rsh.write(f"#SBATCH --mem-per-gpu=40G\n")
    
    rsh.write(f"#SBATCH -o {dir_path}/out_files/out_{save_str}.out\n")
    rsh.write(f"#SBATCH -e {dir_path}/out_files/err_{save_str}.err\n") 
    rsh.write(f"#SBATCH --job-name={save_str}\n")
    rsh.write('''\
nvcc --version
nvidia-smi
pwd
                
which python

''')
    rsh.write(f"export PYTHONPATH=\"${{PYTHONPATH}}:{dir_path}\"\n\n")
    rsh.write(f"{python_loc} vision_tasks.py {flags}\n")

os.system(f"sbatch {bash_file_name}")