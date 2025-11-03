#!/usr/bin/env python3
"""
Migrate preprocessing files from old per-model structure to new dataset-level structure.

Old structure: preprocessing/{MODEL}/{EXPERIMENT}/
New structure: preprocessing/{EXPERIMENT}/

Since preprocessing files are identical across DMPNN, wDMPNN, and DMPNN_DiffPool,
we only need one copy at the dataset level.
"""

import shutil
from pathlib import Path
import argparse


def migrate_preprocessing_files(preprocessing_dir: Path, dry_run: bool = False):
    """Migrate preprocessing files from per-model to dataset-level structure."""
    
    # Models that share identical preprocessing
    models = ['DMPNN', 'wDMPNN', 'DMPNN_DiffPool']
    
    # Track what we've migrated
    migrated = set()
    skipped = set()
    
    print(f"Scanning preprocessing directory: {preprocessing_dir}")
    print(f"Models to check: {', '.join(models)}")
    print()
    
    # Use DMPNN as the source (arbitrary choice since they're identical)
    source_model = 'DMPNN'
    source_dir = preprocessing_dir / source_model
    
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        print("Nothing to migrate.")
        return
    
    # Get all experiment directories from source model
    experiment_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(experiment_dirs)} experiments in {source_model}/")
    print()
    
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        target_dir = preprocessing_dir / exp_name
        
        # Check if already migrated
        if target_dir.exists():
            print(f"⏭️  Skipping {exp_name} - already exists at dataset level")
            skipped.add(exp_name)
            continue
        
        # Check what files exist in this experiment
        files = list(exp_dir.glob('*'))
        if not files:
            print(f"⚠️  Skipping {exp_name} - no files found")
            continue
        
        if dry_run:
            print(f"[DRY RUN] Would migrate {exp_name}:")
            print(f"  Source: {exp_dir}")
            print(f"  Target: {target_dir}")
            print(f"  Files: {len(files)}")
        else:
            # Copy the directory
            try:
                shutil.copytree(exp_dir, target_dir)
                print(f"✅ Migrated {exp_name}")
                print(f"  From: {exp_dir}")
                print(f"  To:   {target_dir}")
                migrated.add(exp_name)
            except Exception as e:
                print(f"❌ Error migrating {exp_name}: {e}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("Migration Summary:")
    print(f"  Migrated: {len(migrated)} experiments")
    print(f"  Skipped:  {len(skipped)} experiments (already exist)")
    
    if dry_run:
        print()
        print("This was a DRY RUN. No files were actually moved.")
        print("Run without --dry-run to perform the migration.")
    else:
        print()
        print("✅ Migration complete!")
        print()
        print("Next steps:")
        print("1. Verify the new structure works with your training scripts")
        print("2. Once verified, you can delete the old per-model directories:")
        for model in models:
            model_dir = preprocessing_dir / model
            if model_dir.exists():
                print(f"   rm -rf {model_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate preprocessing files from per-model to dataset-level structure'
    )
    parser.add_argument('--preprocessing-dir', type=str, default=None,
                       help='Path to preprocessing directory (default: ../../preprocessing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually moving files')
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    if args.preprocessing_dir:
        preprocessing_dir = Path(args.preprocessing_dir)
    else:
        preprocessing_dir = script_dir.parent.parent / "preprocessing"
    
    if not preprocessing_dir.exists():
        print(f"❌ Preprocessing directory not found: {preprocessing_dir}")
        return 1
    
    # Run migration
    migrate_preprocessing_files(preprocessing_dir, dry_run=args.dry_run)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
