"""
Test script to verify PPG integration with your workflow.

Usage:
    python test_ppg_integration.py
"""

import sys
from pathlib import Path

# Add scripts to path
scripts_path = Path(__file__).parent / "scripts" / "python"
sys.path.insert(0, str(scripts_path))

def test_ppg_import():
    """Test that PPG adapter can be imported."""
    print("Testing PPG adapter import...")
    try:
        from chemprop.models import PPGAdapter, create_ppg_args
        print("✅ PPG adapter imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import PPG adapter: {e}")
        return False

def test_ppg_args_creation():
    """Test PPG args creation."""
    print("\nTesting PPG args creation...")
    try:
        from chemprop.models import create_ppg_args
        import argparse
        import numpy as np
        
        # Create mock args
        args = argparse.Namespace()
        args.task_type = 'reg'
        args.model_name = 'PPG'
        args.device = 'cpu'
        
        # Create mock descriptor data
        descriptor_data = np.random.randn(10, 5)
        
        # Create PPG args
        ppg_args = create_ppg_args(args, descriptor_data, n_classes=None)
        
        print(f"✅ PPG args created successfully")
        print(f"   - Dataset type: {ppg_args.dataset_type}")
        print(f"   - Hidden size: {ppg_args.hidden_size}")
        print(f"   - Depth: {ppg_args.depth}")
        print(f"   - Features size: {ppg_args.features_size}")
        return True
    except Exception as e:
        print(f"❌ Failed to create PPG args: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ppg_model_creation():
    """Test PPG model creation through adapter."""
    print("\nTesting PPG model creation...")
    try:
        from chemprop.models import PPGAdapter, create_ppg_args
        import argparse
        import numpy as np
        import torch.nn as nn
        
        # Create mock args
        args = argparse.Namespace()
        args.task_type = 'reg'
        args.model_name = 'PPG'
        args.device = 'cpu'
        
        # Create mock descriptor data
        descriptor_data = np.random.randn(10, 5)
        
        # Create PPG args
        ppg_args = create_ppg_args(args, descriptor_data, n_classes=None)
        
        # Create PPG adapter
        model = PPGAdapter(
            ppg_args=ppg_args,
            output_transform=None,
            loss_function=nn.MSELoss(),
            metric_list=[]
        )
        
        print(f"✅ PPG model created successfully")
        print(f"   - Model type: {type(model).__name__}")
        print(f"   - Task type: {model.task_type}")
        return True
    except Exception as e:
        print(f"❌ Failed to create PPG model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ppg_in_utils():
    """Test that PPG is recognized in utils.py."""
    print("\nTesting PPG in utils.py...")
    try:
        from utils import build_model_and_trainer
        import argparse
        import numpy as np
        
        # Create mock args
        args = argparse.Namespace()
        args.task_type = 'reg'
        args.model_name = 'PPG'
        args.device = 'cpu'
        args.batch_norm = False
        
        # Create mock descriptor data
        descriptor_data = np.random.randn(10, 5)
        
        # Try to build model
        model, trainer = build_model_and_trainer(
            args=args,
            combined_descriptor_data=descriptor_data,
            n_classes=None,
            scaler=None,
            checkpoint_path="test_checkpoint",
            max_epochs=10,
            save_checkpoint=False
        )
        
        print(f"✅ PPG model built through utils successfully")
        print(f"   - Model type: {type(model).__name__}")
        print(f"   - Trainer type: {type(trainer).__name__}")
        return True
    except Exception as e:
        print(f"❌ Failed to build PPG through utils: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("PPG Integration Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Import PPG adapter", test_ppg_import()))
    results.append(("Create PPG args", test_ppg_args_creation()))
    results.append(("Create PPG model", test_ppg_model_creation()))
    results.append(("Build PPG via utils", test_ppg_in_utils()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! PPG integration is working.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
