#!/usr/bin/env python3
"""
Generate training scripts for opv_camb3lyp dataset with DMPNN model.
Creates individual PBS scripts for each target with and without RDKit descriptors.
"""

import os
from pathlib import Path
import argparse

# OPV CAM-B3LYP targets
TARGETS = [
    'optical_lumo', 'gap', 'homo', 'lumo', 'spectral_overlap', 
    'delta_homo', 'delta_lumo', 'delta_optical_lumo', 
    'homo_extrapolated', 'lumo_extrapolated', 'gap_extrapolated', 
    'optical_lumo_extrapolated'
]

# PBS script template
PBS_TEMPLATE = """#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=100GB
#PBS -l walltime={walltime}
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB
#PBS -N DMPNN_opv_camb3lyp_{target}{rdkit_suffix}

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.12.1 cuda/12.0.0
source /home/659/hl4138/dmpnn-venv/bin/activate
cd /scratch/um09/hl4138/dmpnn/


# DMPNN training for target: {target}
python3 scripts/python/train_graph.py --dataset_name opv_camb3lyp --model_name DMPNN --target {target}{rdkit_flag}


##TODO

# Add additional experiments here as needed

"""

def generate_script(target, use_rdkit=False, walltime="04:00:00", output_dir="scripts/shell"):
    """Generate a PBS training script for a specific target."""
    
    rdkit_suffix = "_rdkit" if use_rdkit else ""
    rdkit_flag = " --incl_rdkit" if use_rdkit else ""
    
    script_content = PBS_TEMPLATE.format(
        walltime=walltime,
        target=target,
        rdkit_suffix=rdkit_suffix,
        rdkit_flag=rdkit_flag
    )
    
    filename = f"train_opv_camb3lyp_DMPNN_{target}{rdkit_suffix}.sh"
    filepath = Path(output_dir) / filename
    
    with open(filepath, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(filepath, 0o755)
    
    return filepath

def main():
    parser = argparse.ArgumentParser(description='Generate DMPNN training scripts for opv_camb3lyp')
    parser.add_argument('--output-dir', default='scripts/shell', 
                       help='Output directory for generated scripts')
    parser.add_argument('--walltime', default='36:00:00',
                       help='PBS walltime for each job')
    parser.add_argument('--targets', nargs='+', choices=TARGETS, default=TARGETS,
                       help='Specific targets to generate scripts for')
    parser.add_argument('--rdkit-only', action='store_true',
                       help='Only generate RDKit variants')
    parser.add_argument('--no-rdkit-only', action='store_true', 
                       help='Only generate non-RDKit variants')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be generated without creating files')
    
    args = parser.parse_args()
    
    # Validate conflicting options
    if args.rdkit_only and args.no_rdkit_only:
        print("Error: Cannot specify both --rdkit-only and --no-rdkit-only")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_scripts = []
    
    print(f"🚀 Generating DMPNN training scripts for opv_camb3lyp")
    print(f"📁 Output directory: {output_dir}")
    print(f"⏰ Walltime: {args.walltime}")
    print(f"🎯 Targets: {len(args.targets)} targets")
    print("=" * 60)
    
    for target in args.targets:
        variants = []
        
        # Determine which variants to generate
        if args.rdkit_only:
            variants = [True]
        elif args.no_rdkit_only:
            variants = [False]
        else:
            variants = [False, True]  # Both variants
        
        for use_rdkit in variants:
            variant_name = "with RDKit" if use_rdkit else "without RDKit"
            script_name = f"train_opv_camb3lyp_DMPNN_{target}{'_rdkit' if use_rdkit else ''}.sh"
            
            if args.dry_run:
                print(f"[DRY RUN] Would generate: {script_name} ({variant_name})")
            else:
                try:
                    filepath = generate_script(
                        target=target,
                        use_rdkit=use_rdkit,
                        walltime=args.walltime,
                        output_dir=args.output_dir
                    )
                    generated_scripts.append(filepath)
                    print(f"✅ Generated: {filepath.name} ({variant_name})")
                except Exception as e:
                    print(f"❌ Failed to generate {script_name}: {e}")
    
    print("=" * 60)
    if args.dry_run:
        total_scripts = len(args.targets) * (1 if args.rdkit_only or args.no_rdkit_only else 2)
        print(f"🔍 DRY RUN: Would generate {total_scripts} scripts")
    else:
        print(f"✅ Successfully generated {len(generated_scripts)} training scripts")
        print(f"📂 Scripts saved to: {output_dir}")
        
        if generated_scripts:
            print("\n🚀 To submit all jobs, run:")
            print("cd scripts/shell")
            for script in generated_scripts:
                print(f"qsub {script.name}")
    
    return 0

if __name__ == '__main__':
    exit(main())
