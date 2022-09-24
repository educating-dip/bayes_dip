import argparse
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--dip_file', type=str, default='psnrs_kmnist_dip.yaml')
parser.add_argument('--mcdo_dip_file', type=str, default='results_baseline_kmnist_mcdo_density.yaml')
parser.add_argument('--root_path', type=str, default='.')
parser.add_argument('--save_to', type=str, default='kmnist_psnr_table.tex')
args = parser.parse_args()

with open(os.path.join(args.root_path, args.dip_file), 'r') as f:
    dip_stats = yaml.safe_load(f)

with open(os.path.join(args.root_path, args.mcdo_dip_file), 'r') as f:
    mcdo_dip_stats = yaml.safe_load(f)

NOISE_LIST = [0.05, 0.1]
ANGLES_LIST = [5, 10, 20, 30]

def format_cell_psnr(value):
    return f'${value:.2f}$'
def format_cell_ssim(value):
    return f'${value:.3f}$'

s = ''
s += '\\begin{tabular}{l' + 'rr' + '}\n'
s += ' & PSNR & SSIM\\\\\n'
s += '\\hline\n'

s += 'DIP-MCDO & ' + format_cell_psnr(mcdo_dip_stats['psnr']) + ' & ' + format_cell_ssim(mcdo_dip_stats['ssim']) + '\\\\\n'
s += 'DIP & ' + format_cell_psnr(dip_stats['psnr']) + ' & ' + format_cell_ssim(dip_stats['ssim']) + '\\\\\n'

s += '\\hline\n'
s += '\\end{tabular}\n'

print(s)

if args.save_to:
    with open(args.save_to, 'w') as f:
        f.write(s)
