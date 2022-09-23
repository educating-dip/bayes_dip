import argparse
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--bayes_dip_folder', type=str, default='results_walnut_sample_based_density')
parser.add_argument('--baseline_mcdo_folder', type=str, default='results_baseline_walnut_mcdo_density')
parser.add_argument('--patch_sizes', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--root_path', type=str, default='.')
parser.add_argument('--save_to', type=str, default='walnut_density_table.tex')
args = parser.parse_args()

patch_size_list = args.patch_sizes

bayes_dip_stats = {}
for p in patch_size_list:
    with open(os.path.join(args.root_path, args.bayes_dip_folder, f'patch_size_{p}.yaml'), 'r') as f:
        bayes_dip_stats[p] = yaml.safe_load(f)

baseline_mcdo_stats = {}
for p in patch_size_list:
    with open(os.path.join(args.root_path, args.baseline_mcdo_folder, f'patch_size_{p}.yaml'), 'r') as f:
        baseline_mcdo_stats[p] = yaml.safe_load(f)

s = ''
s += '\\begin{tabular}{l' + 'r' * len(patch_size_list) + '}\n'
s += '& ' + ' & '.join(f'\\shortstack{{${p}\\times {p}$}}' for p in patch_size_list) + '\\\\\n'
s += '\\hline\n'

def format_cell(value):
    return f'${value:.2f}$'

s += 'DIP-MCDO & ' + ' & '.join(format_cell(baseline_mcdo_stats[p]) for p in patch_size_list) + '\\\\\n'
s += 'Bayes DIP (MLL) & ' + ' & '.join(format_cell(bayes_dip_stats[p]['include_predcp_False']) for p in patch_size_list) + '\\\\\n'
s += 'Bayes DIP (TV-MAP) & ' + ' & '.join(format_cell(bayes_dip_stats[p]['include_predcp_True']) for p in patch_size_list) + '\\\\\n'

s += '\\hline\n'
s += '\\end{tabular}\n'

print(s)

if args.save_to:
    with open(args.save_to, 'w') as f:
        f.write(s)
