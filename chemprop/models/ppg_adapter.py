"""
PPG Model Adapter for Chemprop v2.2.0 Integration

This adapter wraps PPG's chemprop v1.4.0 model to work with your v2.2.0 Lightning-based workflow.
It handles the translation between different data formats and APIs.
"""

import sys
from pathlib import Path
from typing import Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule

# Import PPG's chemprop (v1.4.0) using isolated import to avoid circular imports
PPG_AVAILABLE = False
PPGMoleculeModel = None
PPGTrainArgs = None
PPGBatchMolGraph = None

def _import_ppg_isolated():
    """Import PPG modules in isolated way to avoid circular imports."""
    global PPG_AVAILABLE, PPGMoleculeModel, PPGTrainArgs, PPGBatchMolGraph
    
    if PPG_AVAILABLE:
        return True  # Already imported successfully
    
    ppg_root = Path(__file__).parent.parent.parent / "PPG"
    if not ppg_root.exists():
        print(f"Warning: PPG directory not found at {ppg_root}")
        return False
    
    # Save current sys.modules state for chemprop
    original_modules = {}
    chemprop_modules = [k for k in sys.modules.keys() if k.startswith('chemprop')]
    for module in chemprop_modules:
        original_modules[module] = sys.modules[module]
        del sys.modules[module]
    
    # Temporarily add PPG to path
    original_path = sys.path[:]
    sys.path.insert(0, str(ppg_root))
    
    try:
        # Clear any cached chemprop imports
        for module in list(sys.modules.keys()):
            if module.startswith('chemprop'):
                del sys.modules[module]
        
        # Now import PPG's chemprop
        import chemprop.models
        import chemprop.args
        import chemprop.features
        
        PPGMoleculeModel = chemprop.models.MoleculeModel
        PPGTrainArgs = chemprop.args.TrainArgs
        PPGBatchMolGraph = chemprop.features.BatchMolGraph
        
        PPG_AVAILABLE = True
        print("✅ PPG chemprop v1.4.0 imported successfully")
        return True
        
    except Exception as e:
        print(f"Warning: Could not import PPG's chemprop: {e}")
        return False
    
    finally:
        # Restore original sys.path
        sys.path = original_path
        
        # Restore original chemprop modules
        for module, obj in original_modules.items():
            sys.modules[module] = obj

# Try to import PPG when module is loaded
_import_ppg_isolated()


class PPGAdapter(LightningModule):
    """
    Lightning wrapper for PPG's MoleculeModel (chemprop v1.4.0).
    
    This adapter allows PPG models to be trained using your v2.2.0 workflow
    while preserving PPG's original implementation.
    """
    
    def __init__(
        self,
        ppg_args: Any,
        output_transform: Optional[Any] = None,
        loss_function: Optional[nn.Module] = None,
        metric_list: Optional[List] = None
    ):
        """
        Args:
            ppg_args: PPG's TrainArgs object with model configuration
            output_transform: Optional transform to apply to predictions
            loss_function: Loss function to use for training
            metric_list: List of metrics to track
        """
        super().__init__()
        
        if not PPG_AVAILABLE:
            raise ImportError("PPG's chemprop v1.4.0 could not be imported. Check PPG/ directory.")
        
        # Create PPG model
        self.ppg_model = PPGMoleculeModel(ppg_args)
        self.output_transform = output_transform
        self.loss_fn = loss_function
        self.metric_list = metric_list or []
        self.task_type = ppg_args.dataset_type
        
        # Store args for reference
        self.ppg_args = ppg_args
        
    def forward(self, batch, features_batch=None, atom_descriptors_batch=None):
        """
        Forward pass through PPG model.
        
        Args:
            batch: Can be SMILES strings, RDKit mols, or BatchMolGraph
            features_batch: Optional molecular features
            atom_descriptors_batch: Optional atom descriptors
            
        Returns:
            Model predictions
        """
        return self.ppg_model(batch, features_batch, atom_descriptors_batch)
    
    def training_step(self, batch, batch_idx):
        """Lightning training step."""
        bmg, V_d, targets, weights, lt_mask, gt_mask = batch
        
        # PPG expects BatchMolGraph directly
        preds = self(bmg, V_d)
        
        # Apply masks if present
        if lt_mask is not None:
            preds = torch.where(lt_mask, torch.tensor(float('-inf')), preds)
        if gt_mask is not None:
            preds = torch.where(gt_mask, torch.tensor(float('inf')), preds)
        
        # Compute loss
        loss = self.loss_fn(preds, targets, weights)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Lightning validation step."""
        bmg, V_d, targets, weights, lt_mask, gt_mask = batch
        
        preds = self(bmg, V_d)
        
        # Apply output transform if present
        if self.output_transform:
            preds = self.output_transform(preds)
        
        # Apply masks
        if lt_mask is not None:
            preds = torch.where(lt_mask, torch.tensor(float('-inf')), preds)
        if gt_mask is not None:
            preds = torch.where(gt_mask, torch.tensor(float('inf')), preds)
        
        # Compute loss
        loss = self.loss_fn(preds, targets, weights)
        
        self.log('val_loss', loss, prog_bar=True)
        return {'val_loss': loss, 'preds': preds, 'targets': targets}
    
    def test_step(self, batch, batch_idx):
        """Lightning test step."""
        bmg, V_d, targets, weights, lt_mask, gt_mask = batch
        
        preds = self(bmg, V_d)
        
        # Apply output transform
        if self.output_transform:
            preds = self.output_transform(preds)
        
        # Apply masks
        if lt_mask is not None:
            preds = torch.where(lt_mask, torch.tensor(float('-inf')), preds)
        if gt_mask is not None:
            preds = torch.where(gt_mask, torch.tensor(float('inf')), preds)
        
        return {'preds': preds, 'targets': targets}
    
    def predict_step(self, batch, batch_idx):
        """Lightning predict step."""
        bmg, V_d, targets, *_ = batch
        
        preds = self(bmg, V_d)
        
        # Apply output transform
        if self.output_transform:
            preds = self.output_transform(preds)
        
        return preds
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.ppg_args.init_lr if hasattr(self.ppg_args, 'init_lr') else 1e-4
        )
        return optimizer


def create_ppg_args(
    args: Any,
    combined_descriptor_data: Optional[np.ndarray],
    n_classes: Optional[int] = None
) -> Any:
    """
    Convert your args format to PPG's TrainArgs format.
    
    Args:
        args: Your argument namespace
        combined_descriptor_data: Combined RDKit and custom descriptors
        n_classes: Number of classes for classification tasks
        
    Returns:
        PPG TrainArgs object
    """
    # Try to import PPG if not already available
    if not PPG_AVAILABLE:
        if not _import_ppg_isolated():
            raise ImportError("PPG's chemprop v1.4.0 could not be imported.")
    
    # Ensure we have PPG modules
    if PPGMoleculeModel is None or PPGTrainArgs is None:
        raise ImportError("PPG modules not available after import attempt.")
    
    ppg_args = PPGTrainArgs()
    
    # Model architecture
    ppg_args.hidden_size = 300
    ppg_args.depth = 3
    ppg_args.dropout = 0.0
    ppg_args.activation = 'ReLU'
    ppg_args.bias = False
    ppg_args.aggregation = 'mean'
    ppg_args.aggregation_norm = 100
    ppg_args.ffn_num_layers = 2
    ppg_args.ffn_hidden_size = 300
    
    # Task configuration
    if args.task_type == 'reg':
        ppg_args.dataset_type = 'regression'
    elif args.task_type == 'binary':
        ppg_args.dataset_type = 'classification'
    elif n_classes is not None and n_classes > 2:
        ppg_args.dataset_type = 'multiclass'
        ppg_args.multiclass_num_classes = n_classes
    else:
        ppg_args.dataset_type = 'regression'
    
    # Note: num_tasks might be read-only in PPG's TrainArgs
    # Try to set it, but handle the case where it's not writable
    try:
        ppg_args.num_tasks = 1
    except AttributeError as e:
        # If num_tasks is read-only, skip it (PPG will infer from data)
        print(f"Warning: Could not set num_tasks (read-only property): {e}")
    
    # Device configuration
    if hasattr(args, 'device'):
        ppg_args.device = args.device
    else:
        ppg_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Features configuration
    if combined_descriptor_data is not None:
        ppg_args.use_input_features = True
        ppg_args.features_size = combined_descriptor_data.shape[1]
    else:
        ppg_args.use_input_features = False
        ppg_args.features_size = 0
    
    ppg_args.features_only = False
    ppg_args.atom_messages = False
    ppg_args.undirected = False
    ppg_args.number_of_molecules = 1
    
    # Training configuration
    ppg_args.init_lr = 1e-4
    ppg_args.max_lr = 1e-3
    ppg_args.final_lr = 1e-4
    
    # Atom descriptors
    ppg_args.atom_descriptors = None
    ppg_args.overwrite_default_atom_features = False
    ppg_args.overwrite_default_bond_features = False
    
    # Checkpoint/freezing
    ppg_args.checkpoint_frzn = None
    ppg_args.freeze_first_only = False
    ppg_args.frzn_ffn_layers = 0
    
    # MPN sharing
    ppg_args.mpn_shared = False
    
    return ppg_args
