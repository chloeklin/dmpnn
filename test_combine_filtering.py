#!/usr/bin/env python3
"""
Test script to verify OPV target filtering in combine_target_results.py
"""
import sys
sys.path.append('scripts/python')
from combine_target_results import combine_results
import tempfile
import pandas as pd
from pathlib import Path

def test_opv_filtering():
    """Test that OPV targets are properly filtered."""
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files with different targets
        test_data = pd.DataFrame({
            'mae': [0.1, 0.2],
            'r2': [0.8, 0.9],
            'split': [0, 1]
        })
        
        # Allowed OPV target
        allowed_file = temp_path / "opv_camb3lyp_results_gap.csv"
        test_data.to_csv(allowed_file, index=False)
        
        # Disallowed OPV target  
        disallowed_file = temp_path / "opv_camb3lyp_results_delta_homo.csv"
        test_data.to_csv(disallowed_file, index=False)
        
        # Non-OPV target (should be kept)
        non_opv_file = temp_path / "tc_results_TC.csv"
        test_data.to_csv(non_opv_file, index=False)
        
        print("Created test files:")
        for f in temp_path.glob("*.csv"):
            print(f"  {f.name}")
        
        # Run combine_results
        print("\nRunning combine_results...")
        combine_results(str(temp_path))
        
        # Check results
        print("\nFinal files:")
        for f in temp_path.glob("*.csv"):
            print(f"  {f.name}")
            
        # Check if filtering worked
        opv_combined = temp_path / "opv_camb3lyp_results.csv"
        if opv_combined.exists():
            df = pd.read_csv(opv_combined)
            targets = df['target'].unique() if 'target' in df.columns else []
            print(f"\nOPV combined targets: {list(targets)}")
            
            # Should only contain 'gap', not 'delta_homo'
            if 'gap' in targets and 'delta_homo' not in targets:
                print("✅ OPV filtering working correctly!")
            else:
                print("❌ OPV filtering not working as expected")
        else:
            print("❌ No OPV combined file created")

if __name__ == "__main__":
    test_opv_filtering()
