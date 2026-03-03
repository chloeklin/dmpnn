#!/usr/bin/env python3
"""Analyze embeddings directory to find complete sets (all 5 splits)"""

import os
from pathlib import Path
from collections import defaultdict
import re

embeddings_dir = Path('/Users/u6788552/Desktop/experiments/dmpnn/results/embeddings')

# Parse all embedding files
embeddings = defaultdict(lambda: defaultdict(set))

for file in embeddings_dir.glob('*.npy'):
    # Extract split number
    match = re.search(r'split_(\d+)\.npy$', file.name)
    if not match:
        continue
    split_num = int(match.group(1))
    
    # Remove the split suffix to get the base pattern
    base = re.sub(r'_(X_train|X_val|X_test|feature_mask)_split_\d+\.npy$', '', file.name)
    
    # Parse: dataset__model__target__[variants]
    parts = base.split('__')
    if len(parts) < 3:
        continue
    
    dataset = parts[0]
    model = parts[1]
    target = parts[2]
    
    # Collect variants (desc, rdkit, size, etc.)
    variants = '__'.join(parts[3:]) if len(parts) > 3 else ''
    
    key = f'{dataset}__{model}__{target}'
    if variants:
        key += f'__{variants}'
    
    embeddings[dataset][key].add(split_num)

# Report complete sets (all 5 splits: 0,1,2,3,4)
print('=' * 80)
print('MODELS WITH COMPLETE EMBEDDINGS (all 5 splits)')
print('=' * 80)

for dataset in sorted(embeddings.keys()):
    complete_configs = []
    incomplete_configs = []
    
    for config, splits in sorted(embeddings[dataset].items()):
        if splits == {0, 1, 2, 3, 4}:
            complete_configs.append(config)
        else:
            incomplete_configs.append((config, sorted(splits)))
    
    if complete_configs or incomplete_configs:
        print(f'\n{dataset}:')
        
        if complete_configs:
            print(f'  Complete ({len(complete_configs)} configurations):')
            for config in complete_configs:
                # Extract just model and variants for display
                parts = config.split('__')
                model = parts[1]
                target = parts[2]
                variants = '__'.join(parts[3:]) if len(parts) > 3 else 'base'
                print(f'     - {model} / {target} / {variants}')
        
        if incomplete_configs:
            print(f'  Incomplete ({len(incomplete_configs)} configurations):')
            for config, splits in incomplete_configs:
                parts = config.split('__')
                model = parts[1]
                target = parts[2]
                variants = '__'.join(parts[3:]) if len(parts) > 3 else 'base'
                missing = sorted(set(range(5)) - set(splits))
                print(f'     - {model} / {target} / {variants} (has {splits}, missing {missing})')

print('\n' + '=' * 80)
print('\nSUMMARY:')
total_complete = sum(len([c for c, s in embeddings[d].items() if s == {0,1,2,3,4}]) for d in embeddings)
total_incomplete = sum(len([c for c, s in embeddings[d].items() if s != {0,1,2,3,4}]) for d in embeddings)
print(f'Total complete configurations: {total_complete}')
print(f'Total incomplete configurations: {total_incomplete}')
print('=' * 80)
