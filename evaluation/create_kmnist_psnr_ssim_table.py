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

def format_cell(stats):
    return f'${stats["psnr"]["mean"]:.2f}$/${stats["ssim"]["mean"]:.3f}$'

all_tables = ''

for noise in NOISE_LIST:
    s = ''
    s += '\\begin{tabular}{l' + 'r' * len(ANGLES_LIST) + '}\n'
    s += '\\textbf{' + f'{noise * 100:.0f}' + '\\% white noise}&\\hspace{-2cm}$\\#$directions:\\quad ' + ' & '.join(f'${a}$' for a in ANGLES_LIST) + '\\\\\n'
    s += '\\hline\n'

    s += 'DIP-MCDO & ' + ' & '.join(format_cell(mcdo_dip_stats[noise][a]) for a in ANGLES_LIST) + '\\\\\n'
    s += 'DIP & ' + ' & '.join(format_cell(dip_stats[noise][a]) for a in ANGLES_LIST) + '\\\\\n'

    s += '\\hline\n'
    s += '\\end{tabular}\n'

    print(f'\ntable for noise={noise}:\n')
    print(s)

    all_tables += s

if args.save_to:
    with open(args.save_to, 'w') as f:
        f.write(all_tables)
