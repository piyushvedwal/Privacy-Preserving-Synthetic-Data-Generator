# model_train.py
# Implements CTGAN with Differential Privacy integration using Opacus

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ctgan import CTGAN
from dp_gan import DPGAN
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
import time
import os
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PrivacyPreservingCTGAN:
    """CTGAN model with optional Differential Privacy integration."""
    
    def __init__(self, 
                 use_dp: bool = False, 
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 max_grad_norm: float = 1.0,
                 epochs: int = 300,
                 batch_size: int = 500,
                 verbose: bool = True):
        """Initialize the model.
        
        Args:
            use_dp: Whether to use Differential Privacy
            epsilon: Privacy budget (lower = more private)
            delta: Failure probability (typically 1/N where N is dataset size)
            max_grad_norm: Maximum norm of gradients for clipping
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print training progress
        """
        self.use_dp = use_dp
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.privacy_engine = None
        self.final_epsilon = None
        self.final_alpha = None
        self.privacy_report = {}
        
    def train(self, data: pd.DataFrame, discrete_columns: List[str] = None,
              column_groups: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """Train the synthesizer with or without Differential Privacy.
        
        Without DP this trains a standard CTGAN. With DP it trains a DP-GAN whose
        discriminator is the only component that sees real data and is trained with
        DP-SGD (Opacus); see dp_gan.py for the privacy argument.
        
        Args:
            data: Input DataFrame (preprocessed)
            discrete_columns: List of discrete column names
            column_groups: Optional mapping of original categorical column -> its one-hot columns
            
        Returns:
            Dictionary with training metrics and privacy budget
        """
        start_time = time.time()
        
        # Initialize CTGAN model
        if discrete_columns is None:
            discrete_columns = []
            
        # Configure CTGAN parameters
        ctgan_params = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'verbose': self.verbose
        }
        
        if self.use_dp:
            logging.info(f"Training DP-GAN with Differential Privacy (target ε={self.epsilon}, δ={self.delta})")
            self.model = DPGAN(
                epsilon=self.epsilon,
                delta=self.delta,
                max_grad_norm=self.max_grad_norm,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )
            self.privacy_report = self.model.fit(data, discrete_columns, column_groups)
            self.final_epsilon = self.privacy_report["epsilon"]
            self.final_alpha = self.privacy_report["alpha"]
            
            logging.info(f"Final privacy guarantee: (ε={self.final_epsilon:.2f}, δ={self.delta})")
        else:
            logging.info("Training standard CTGAN (no privacy guarantees)")
            # Standard training without DP
            self.model = CTGAN(
                embedding_dim=128,
                generator_dim=(256, 256),
                discriminator_dim=(256, 256),
                **ctgan_params
            )
            self.model.fit(data, discrete_columns)
        
        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time:.2f} seconds")
        
        # Return training metrics
        metrics = {
            "training_time": training_time,
            "epochs": self.epochs,
            "use_dp": self.use_dp,
        }
        
        if self.use_dp:
            metrics["epsilon"] = self.final_epsilon
            metrics["delta"] = self.delta
            metrics["alpha"] = self.final_alpha
            metrics["noise_multiplier"] = self.privacy_report["noise_multiplier"]
            metrics["sample_rate"] = self.privacy_report["sample_rate"]
            metrics["expected_batch_size"] = self.privacy_report["expected_batch_size"]
            metrics["steps"] = self.privacy_report["steps"]
        
        return metrics
    
    def generate(self, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data.
        
        Args:
            num_rows: Number of synthetic rows to generate
            
        Returns:
            DataFrame with synthetic data
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        logging.info(f"Generating {num_rows} synthetic rows")
        synthetic_data = self.model.sample(num_rows)
        
        # Clean up any None values in the generated data
        for col in synthetic_data.columns:
            if synthetic_data[col].isna().any():
                # Replace NaN values with column mean for numerical columns
                if pd.api.types.is_numeric_dtype(synthetic_data[col]):
                    col_mean = synthetic_data[col].mean()
                    synthetic_data[col] = synthetic_data[col].fillna(col_mean)
                # Replace NaN values with most frequent value for non-numerical columns
                else:
                    col_mode = synthetic_data[col].mode()[0] if not synthetic_data[col].mode().empty else "Unknown"
                    synthetic_data[col] = synthetic_data[col].fillna(col_mode)
        
        return synthetic_data
    
    def save_model(self, path: str) -> None:
        """Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save")
            
        torch.save(self.model, path)
        logging.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        self.model = torch.load(path, weights_only=False)
        logging.info(f"Model loaded from {path}")