import argparse
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--bayes_dip_file', type=str, default='results_kmnist_exact_density.yaml')
parser.add_argument('--baseline_deterministic_file', type=str, default='results_baseline_kmnist_deterministic_density.yaml')
parser.add_argument('--baseline_mcdo_file', type=str, default='results_baseline_kmnist_mcdo_density.yaml')
parser.add_argument('--root_path', type=str, default='.')
parser.add_argument('--save_to', type=str, default='kmnist_density_table.tex')
args = parser.parse_args()

with open(os.path.join(args.root_path, args.bayes_dip_file), 'r') as f:
    bayes_dip_stats = yaml.safe_load(f)

with open(os.path.join(args.root_path, args.baseline_deterministic_file), 'r') as f:
    baseline_deterministic_stats = yaml.safe_load(f)

with open(os.path.join(args.root_path, args.baseline_mcdo_file), 'r') as f:
    baseline_mcdo_stats = yaml.safe_load(f)

NOISE_LIST = [0.05, 0.1]
ANGLES_LIST = [5, 10, 20, 30]

def format_cell(stats):
    return f'${stats["mean"]:.2f} \\pm {stats["stderr"]:.2f}$'

all_tables = ''

for noise in NOISE_LIST:
    s = ''
    s += '\\begin{tabular}{l' + 'r' * len(ANGLES_LIST) + '}\n'
    s += '\\textbf{' + f'{noise * 100:.0f}' + '\\% white noise}&\\hspace{-2cm}$\\#$directions:\\quad ' + ' & '.join(f'${a}$' for a in ANGLES_LIST) + '\\\\\n'
    s += '\\hline\n'

    s += 'DIP ($\\sigma^2_y$ = 1) & ' + ' & '.join(format_cell(baseline_deterministic_stats[noise][a]['load_log_noise_variance_False']) for a in ANGLES_LIST) + '\\\\\n'
    s += 'DIP (MLL $\\sigma^2_y$) & ' + ' & '.join(format_cell(baseline_deterministic_stats[noise][a]['load_log_noise_variance_True']) for a in ANGLES_LIST) + '\\\\\n'
    s += 'DIP-MCDO & ' + ' & '.join(format_cell(baseline_mcdo_stats[noise][a]) for a in ANGLES_LIST) + '\\\\\n'
    s += 'Bayes DIP (MLL) & ' + ' & '.join(format_cell(bayes_dip_stats[noise][a]['include_predcp_False']) for a in ANGLES_LIST) + '\\\\\n'
    s += 'Bayes DIP (TV-MAP) & ' + ' & '.join(format_cell(bayes_dip_stats[noise][a]['include_predcp_True']) for a in ANGLES_LIST) + '\\\\\n'

    s += '\\hline\n'
    s += '\\end{tabular}\n'

    print(f'\ntable for noise={noise}:\n')
    print(s)

    all_tables += s

if args.save_to:
    with open(args.save_to, 'w') as f:
        f.write(all_tables)
