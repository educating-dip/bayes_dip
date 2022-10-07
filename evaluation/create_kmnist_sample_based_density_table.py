import argparse
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--bayes_dip_folder', type=str, default='results_kmnist_sample_based_density')
parser.add_argument('--baseline_mcdo_folder', type=str, default='results_baseline_kmnist_mcdo_density')
parser.add_argument('--noise_list', type=float, nargs='+', default=[0.05, 0.1])
parser.add_argument('--angles_list', type=int, nargs='+', default=[20])
parser.add_argument('--patch_sizes', type=int, nargs='+', default=list(range(1, 29)))
parser.add_argument('--root_path', type=str, default='.')
parser.add_argument('--save_to', type=str, default='kmnist_sample_based_density_table.tex')
args = parser.parse_args()

patch_size_list = args.patch_sizes

bayes_dip_stats = {}
for p in patch_size_list:
    with open(os.path.join(args.root_path, args.bayes_dip_folder, f'patch_size_{p}.yaml'), 'r') as f:
        bayes_dip_stats[p] = yaml.safe_load(f)

baseline_mcdo_stats = {}
if args.baseline_mcdo_folder:
    for p in patch_size_list:
        with open(os.path.join(args.root_path, args.baseline_mcdo_folder, f'patch_size_{p}.yaml'), 'r') as f:
            baseline_mcdo_stats[p] = yaml.safe_load(f)

def format_cell(stats):
    return f'${stats["mean"]:.2f} \\pm {stats["stderr"]:.2f}$'

all_tables = ''

for noise in args.noise_list:
    for angles in args.angles_list:
        s = ''
        s += '\\textbf{' + f'{noise * 100:.0f}' + '\\% white noise, ' + f'{angles}' + ' directions}\n'
        s += '\\begin{tabular}{l' + 'r' * len(patch_size_list) + '}\n'
        s += '& ' + ' & '.join(f'\\shortstack{{${p}\\times {p}$}}' for p in patch_size_list) + '\\\\\n'
        s += '\\hline\n'

        if baseline_mcdo_stats:
            s += 'DIP-MCDO & ' + ' & '.join(format_cell(baseline_mcdo_stats[p][noise][angles]) for p in patch_size_list) + '\\\\\n'
        s += 'Bayes DIP (MLL) & ' + ' & '.join(format_cell(bayes_dip_stats[p][noise][angles]['include_predcp_False']) for p in patch_size_list) + '\\\\\n'

        s += '\\hline\n'
        s += '\\end{tabular}\n'

        print(f'\ntable for noise={noise}, angles={angles}:\n')
        print(s)

        all_tables += s

if args.save_to:
    with open(args.save_to, 'w') as f:
        f.write(all_tables)
