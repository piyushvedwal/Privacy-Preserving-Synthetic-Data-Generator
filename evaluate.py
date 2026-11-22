# evaluate.py
# Implements metrics for evaluating synthetic data quality and privacy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, classification_report
from typing import Dict, List, Tuple, Union, Optional, Any
import logging

from privacy_checks import distance_membership_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def encode_features(frames: List[pd.DataFrame], columns: List[str]) -> List[np.ndarray]:
    """Encode several frames identically for scikit-learn models.

    Numeric columns are kept (missing values filled with the median across frames);
    everything else is one-hot encoded with categories from all frames, so text
    columns such as labels no longer break the classifiers.
    """
    keys = [str(i) for i in range(len(frames))]
    combined = pd.concat([f[columns] for f in frames], keys=keys)
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(combined[c])]
    other_cols = [c for c in columns if c not in numeric_cols]
    for c in numeric_cols:
        combined[c] = pd.to_numeric(combined[c], errors="coerce")
        combined[c] = combined[c].fillna(combined[c].median() if combined[c].notna().any() else 0)
    if other_cols:
        combined[other_cols] = combined[other_cols].astype(str)
        combined = pd.get_dummies(combined, columns=other_cols, dtype=float)
    return [combined.loc[k].to_numpy(dtype=float) for k in keys]


class SyntheticDataEvaluator:
    """Evaluates synthetic data quality and privacy."""
    
    def __init__(self):
        self.metrics = {}
        self.column_types = {}
    
    def set_column_types(self, column_types: Dict[str, str]) -> None:
        """Set column types for evaluation.
        
        Args:
            column_types: Dictionary mapping column names to types ('numerical' or 'categorical')
        """
        self.column_types = column_types
    
    def evaluate_fidelity(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate how well synthetic data preserves statistical properties of real data.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            
        Returns:
            Dictionary of fidelity metrics
        """
        logging.info("Evaluating fidelity metrics...")
        fidelity_metrics = {}
        
        # Ensure we're only comparing columns that exist in both datasets
        common_columns = list(set(real_data.columns) & set(synthetic_data.columns))
        
        # If column types weren't explicitly set, detect them
        if not self.column_types:
            self.column_types = {}
            for col in common_columns:
                if pd.api.types.is_numeric_dtype(real_data[col]):
                    self.column_types[col] = 'numerical'
                else:
                    self.column_types[col] = 'categorical'
        
        # 1. Kolmogorov-Smirnov test for numerical columns
        ks_scores = {}
        for col in common_columns:
            if col in self.column_types and self.column_types[col] == 'numerical':
                try:
                    # Drop NaN values for KS test
                    real_col = real_data[col].dropna()
                    synth_col = synthetic_data[col].dropna()
                    
                    if len(real_col) > 0 and len(synth_col) > 0:
                        ks_statistic, ks_pvalue = stats.ks_2samp(real_col, synth_col)
                        # Convert to similarity score (1 - statistic)
                        ks_scores[col] = 1.0 - ks_statistic
                except Exception as e:
                    logging.warning(f"Could not compute KS test for column {col}: {e}")
                    ks_scores[col] = None
        
        # Average KS similarity score
        valid_ks_scores = [score for score in ks_scores.values() if score is not None]
        if valid_ks_scores:
            fidelity_metrics['ks_mean'] = np.mean(valid_ks_scores)
            fidelity_metrics['ks_scores'] = ks_scores
        else:
            fidelity_metrics['ks_mean'] = None
            fidelity_metrics['ks_scores'] = {}
        
        # 2. Chi-square test for categorical columns
        chi2_scores = {}
        for col in common_columns:
            if col in self.column_types and self.column_types[col] == 'categorical':
                try:
                    # Get value counts
                    real_counts = real_data[col].value_counts(normalize=True)
                    synth_counts = synthetic_data[col].value_counts(normalize=True)
                    
                    # Align the indices
                    all_values = list(set(real_counts.index) | set(synth_counts.index))
                    real_counts = real_counts.reindex(all_values, fill_value=0)
                    synth_counts = synth_counts.reindex(all_values, fill_value=0)
                    
                    # Chi-square test
                    chi2_stat, chi2_pvalue = stats.chisquare(
                        f_obs=synth_counts * len(synthetic_data),
                        f_exp=real_counts * len(synthetic_data)
                    )
                    
                    # Convert to similarity score (using p-value)
                    chi2_scores[col] = chi2_pvalue
                except Exception as e:
                    logging.warning(f"Could not compute Chi-square test for column {col}: {e}")
                    chi2_scores[col] = None
        
        # Category similarity: 1 - total variation distance between category frequencies.
        # Unlike a chi-square p-value, this does not collapse towards 0 as row counts grow.
        tvd_scores = {}
        for col in common_columns:
            if self.column_types.get(col) == 'categorical':
                real_freq = real_data[col].astype(str).value_counts(normalize=True)
                synth_freq = synthetic_data[col].astype(str).value_counts(normalize=True)
                values = real_freq.index.union(synth_freq.index)
                tvd = 0.5 * np.abs(real_freq.reindex(values, fill_value=0) - synth_freq.reindex(values, fill_value=0)).sum()
                tvd_scores[col] = 1.0 - float(tvd)
        fidelity_metrics['category_similarity_scores'] = tvd_scores
        fidelity_metrics['category_similarity_mean'] = float(np.mean(list(tvd_scores.values()))) if tvd_scores else None
        
        # Average Chi-square p-value (kept for reference; not used in the overall score)
        valid_chi2_scores = [score for score in chi2_scores.values() if score is not None]
        if valid_chi2_scores:
            fidelity_metrics['chi2_mean'] = np.mean(valid_chi2_scores)
            fidelity_metrics['chi2_scores'] = chi2_scores
        else:
            fidelity_metrics['chi2_mean'] = None
            fidelity_metrics['chi2_scores'] = {}
        
        # 3. Correlation similarity
        try:
            # Calculate correlation matrices
            numerical_cols = [col for col in common_columns 
                             if col in self.column_types and self.column_types[col] == 'numerical']
            
            if len(numerical_cols) >= 2:
                real_corr = real_data[numerical_cols].corr().values.flatten()
                synth_corr = synthetic_data[numerical_cols].corr().values.flatten()
                
                # Calculate correlation similarity (using Pearson correlation between correlation matrices)
                corr_similarity, _ = stats.pearsonr(real_corr, synth_corr)
                fidelity_metrics['correlation_similarity'] = corr_similarity
            else:
                fidelity_metrics['correlation_similarity'] = None
        except Exception as e:
            logging.warning(f"Could not compute correlation similarity: {e}")
            fidelity_metrics['correlation_similarity'] = None
        
        # 4. Propensity mean squared error (pMSE)
        try:
            # Create a combined dataset with a label indicating real (1) or synthetic (0)
            real_labeled = real_data.copy()
            real_labeled['is_real'] = 1
            synth_labeled = synthetic_data.copy()
            synth_labeled['is_real'] = 0
            
            # Combine datasets
            combined_data = pd.concat([real_labeled, synth_labeled], axis=0)
            
            # Encode common columns (one-hot for text columns) for classification
            X = encode_features([combined_data], common_columns)[0]
            y = combined_data['is_real'].to_numpy()
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Train a classifier to distinguish real from synthetic
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            clf.fit(X_train, y_train)
            
            # Predict probabilities
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            
            # Calculate AUC
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Convert AUC to a similarity score (0 = perfectly distinguishable, 1 = indistinguishable).
            # AUC below 0.5 means the classifier does no better than chance, so it counts as 1.
            pmse_similarity = 1.0 - 2.0 * max(0.0, auc - 0.5)
            fidelity_metrics['pmse_similarity'] = pmse_similarity
            fidelity_metrics['real_vs_synthetic_auc'] = auc
        except Exception as e:
            logging.warning(f"Could not compute propensity score: {e}")
            fidelity_metrics['pmse_similarity'] = None
            fidelity_metrics['real_vs_synthetic_auc'] = None
        
        # 5. Overall fidelity score (average of available metrics)
        available_scores = []
        if fidelity_metrics['ks_mean'] is not None:
            available_scores.append(fidelity_metrics['ks_mean'])
        if fidelity_metrics['category_similarity_mean'] is not None:
            available_scores.append(fidelity_metrics['category_similarity_mean'])
        if fidelity_metrics['correlation_similarity'] is not None:
            available_scores.append(fidelity_metrics['correlation_similarity'])
        if fidelity_metrics['pmse_similarity'] is not None:
            available_scores.append(fidelity_metrics['pmse_similarity'])
        
        if available_scores:
            fidelity_metrics['overall_fidelity'] = np.mean(available_scores)
        else:
            fidelity_metrics['overall_fidelity'] = None
        
        logging.info(f"Fidelity evaluation complete. Overall score: {fidelity_metrics['overall_fidelity']}")
        return fidelity_metrics
    
    def evaluate_privacy(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                         privacy_metrics: Optional[Dict[str, Any]] = None,
                         train_data: Optional[pd.DataFrame] = None,
                         holdout_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Evaluate privacy protection of synthetic data.
        
        Membership inference needs real records the generator never saw. When
        train_data (rows the generator was trained on) and holdout_data (rows held
        out of training) are given, a distance-based membership inference attack is
        run; otherwise the attack is skipped rather than reporting a misleading score.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            privacy_metrics: Optional dictionary with privacy guarantees (epsilon, delta)
            train_data: Real rows used to train the generator
            holdout_data: Real rows held out of training
            
        Returns:
            Dictionary of privacy metrics
        """
        logging.info("Evaluating privacy metrics...")
        privacy_results = {}
        
        # Include formal privacy guarantees if provided
        if privacy_metrics is not None:
            if 'epsilon' in privacy_metrics:
                privacy_results['epsilon'] = privacy_metrics['epsilon']
            if 'delta' in privacy_metrics:
                privacy_results['delta'] = privacy_metrics['delta']
        
        privacy_results['membership_inference_accuracy'] = None
        privacy_results['membership_inference_auc'] = None
        privacy_results['privacy_score'] = None
        
        if train_data is None or holdout_data is None:
            privacy_results['membership_inference_note'] = (
                "Skipped: membership inference needs rows held out of training.")
            logging.info("Skipping membership inference: no held-out rows available.")
            return privacy_results
        
        try:
            mia = distance_membership_inference(train_data, holdout_data, synthetic_data)
            auc = mia['membership_inference_auc']
            privacy_results.update(mia)
            # 1 = attacker no better than chance, 0 = attacker always right
            privacy_results['privacy_score'] = 1.0 - max(0.0, auc - 0.5) * 2
            privacy_results['membership_inference_note'] = (
                f"Distance-based attack on {mia['records_per_group']} training vs "
                f"{mia['records_per_group']} held-out rows.")
            logging.info(f"Membership inference results - AUC: {auc:.4f}, "
                         f"Accuracy: {mia['membership_inference_accuracy']:.4f}")
        except Exception as e:
            logging.warning(f"Could not compute membership inference attack: {e}")
            privacy_results['membership_inference_note'] = f"Failed: {e}"
        
        return privacy_results
    
    def evaluate_utility(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                         target_column: str,
                         train_data: Optional[pd.DataFrame] = None,
                         holdout_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Evaluate utility of synthetic data for machine learning tasks.
        
        Train-on-synthetic, test-on-real: one model is trained on real rows, one on
        synthetic rows, and both are scored on the same real test rows. With held-out
        rows the test set is data the generator never saw; otherwise a split of
        real_data is used, which flatters the synthetic model.
        
        Classification targets are scored by accuracy; numeric targets with many
        distinct values are treated as regression and scored by R².
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            target_column: Target column for prediction task
            train_data: Real rows used to train the generator
            holdout_data: Real rows held out of training
            
        Returns:
            Dictionary of utility metrics
        """
        logging.info(f"Evaluating utility metrics for target column: {target_column}")
        utility_metrics = {
            'real_model_accuracy': None,
            'synthetic_model_accuracy': None,
            'utility_ratio': None,
        }
        
        try:
            # Ensure target column exists in both datasets
            if target_column not in real_data.columns or target_column not in synthetic_data.columns:
                raise ValueError(f"Target column {target_column} not found in both datasets")
            
            feature_columns = [col for col in real_data.columns
                               if col != target_column and col in synthetic_data.columns]
            
            if train_data is not None and holdout_data is not None:
                real_train, real_test = train_data, holdout_data
                utility_metrics['evaluated_on'] = 'held-out real rows'
            else:
                real_train, real_test = train_test_split(real_data, test_size=0.2, random_state=42)
                utility_metrics['evaluated_on'] = 'split of the training data (optimistic)'
            
            synth = synthetic_data.dropna(subset=[target_column])
            real_train = real_train.dropna(subset=[target_column])
            real_test = real_test.dropna(subset=[target_column])
            
            target = real_data[target_column]
            is_regression = (pd.api.types.is_numeric_dtype(target)
                             and self.column_types.get(target_column, 'numerical') == 'numerical'
                             and target.nunique() > 20)
            
            X_real_train, X_real_test, X_synth = encode_features(
                [real_train, real_test, synth], feature_columns)
            
            if is_regression:
                y = lambda d: pd.to_numeric(d[target_column], errors='coerce').to_numpy()
                model_cls, score_fn, score_name = RandomForestRegressor, r2_score, 'R²'
            else:
                y = lambda d: d[target_column].astype(str).to_numpy()
                model_cls, score_fn, score_name = RandomForestClassifier, accuracy_score, 'accuracy'
            
            real_model = model_cls(n_estimators=100, random_state=42).fit(X_real_train, y(real_train))
            real_score = score_fn(y(real_test), real_model.predict(X_real_test))
            
            synth_model = model_cls(n_estimators=100, random_state=42).fit(X_synth, y(synth))
            synth_score = score_fn(y(real_test), synth_model.predict(X_real_test))
            
            utility_ratio = max(synth_score, 0.0) / real_score if real_score > 0 else None
            
            utility_metrics.update({
                'task': 'regression' if is_regression else 'classification',
                'score_name': score_name,
                'real_model_accuracy': float(real_score),
                'synthetic_model_accuracy': float(synth_score),
                'utility_ratio': utility_ratio,
            })
            logging.info(f"Utility evaluation complete ({score_name}). Ratio: {utility_ratio}")
        except Exception as e:
            logging.warning(f"Could not compute utility metrics: {e}")
            utility_metrics['error'] = str(e)
        
        return utility_metrics
    
    def plot_distributions(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                           columns: Optional[List[str]] = None, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Plot distributions of real vs synthetic data.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            columns: Optional list of columns to plot (defaults to all numerical columns)
            figsize: Figure size
            
        Returns:
            Matplotlib figure with distribution plots
        """
        # Ensure we're only comparing columns that exist in both datasets
        common_columns = list(set(real_data.columns) & set(synthetic_data.columns))
        
        # If no columns specified, use all numerical columns
        if columns is None:
            columns = [col for col in common_columns if pd.api.types.is_numeric_dtype(real_data[col])]
        else:
            # Filter to ensure all requested columns exist in both datasets
            columns = [col for col in columns if col in common_columns]
        
        # Determine grid size based on number of columns
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each column
        for i, col in enumerate(columns):
            if i < len(axes):
                ax = axes[i]
                
                if pd.api.types.is_numeric_dtype(real_data[col]):
                    # Numerical column - use histograms
                    sns.histplot(real_data[col], color='blue', alpha=0.5, label='Real', ax=ax, kde=True)
                    sns.histplot(synthetic_data[col], color='red', alpha=0.5, label='Synthetic', ax=ax, kde=True)
                else:
                    # Categorical column - use bar plots
                    real_counts = real_data[col].value_counts(normalize=True)
                    synth_counts = synthetic_data[col].value_counts(normalize=True)
                    
                    # Combine categories
                    all_cats = list(set(real_counts.index) | set(synth_counts.index))
                    real_counts = real_counts.reindex(all_cats, fill_value=0)
                    synth_counts = synth_counts.reindex(all_cats, fill_value=0)
                    
                    # Create DataFrame for plotting
                    plot_df = pd.DataFrame({
                        'Real': real_counts,
                        'Synthetic': synth_counts
                    })
                    
                    # Plot
                    plot_df.plot(kind='bar', ax=ax, alpha=0.7)
                
                ax.set_title(col)
                ax.legend()
                ax.set_ylabel('Frequency')
                
                # Rotate x-axis labels for categorical variables
                if not pd.api.types.is_numeric_dtype(real_data[col]):
                    ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(columns), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_comparison(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                                   figsize: Tuple[int, int] = (15, 7)) -> plt.Figure:
        """Plot correlation matrices for real and synthetic data.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            figsize: Figure size
            
        Returns:
            Matplotlib figure with correlation heatmaps
        """
        # Get numerical columns that exist in both datasets
        common_columns = list(set(real_data.columns) & set(synthetic_data.columns))
        numerical_cols = [col for col in common_columns if pd.api.types.is_numeric_dtype(real_data[col])]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot correlation matrices
        if len(numerical_cols) >= 2:
            # Real data correlation
            real_corr = real_data[numerical_cols].corr()
            sns.heatmap(real_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax1, vmin=-1, vmax=1)
            ax1.set_title('Real Data Correlation')
            
            # Synthetic data correlation
            synth_corr = synthetic_data[numerical_cols].corr()
            sns.heatmap(synth_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax2, vmin=-1, vmax=1)
            ax2.set_title('Synthetic Data Correlation')
        else:
            ax1.text(0.5, 0.5, 'Not enough numerical columns\nfor correlation analysis', 
                     ha='center', va='center', fontsize=12)
            ax2.text(0.5, 0.5, 'Not enough numerical columns\nfor correlation analysis', 
                     ha='center', va='center', fontsize=12)
            ax1.set_title('Real Data')
            ax2.set_title('Synthetic Data')
        
        plt.tight_layout()
        return fig
    
    def evaluate_all(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                     privacy_metrics: Optional[Dict[str, Any]] = None,
                     target_column: Optional[str] = None,
                     train_data: Optional[pd.DataFrame] = None,
                     holdout_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run all evaluations and return combined results.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            privacy_metrics: Optional dictionary with privacy guarantees
            target_column: Optional target column for utility evaluation
            train_data: Real rows used to train the generator (enables membership inference)
            holdout_data: Real rows held out of training (enables membership inference)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # Fidelity metrics
        fidelity_metrics = self.evaluate_fidelity(real_data, synthetic_data)
        results['fidelity'] = fidelity_metrics
        
        # Privacy metrics
        privacy_metrics = self.evaluate_privacy(real_data, synthetic_data, privacy_metrics,
                                                train_data, holdout_data)
        results['privacy'] = privacy_metrics
        
        # Utility metrics (if target column provided)
        if target_column is not None and target_column in real_data.columns and target_column in synthetic_data.columns:
            utility_metrics = self.evaluate_utility(real_data, synthetic_data, target_column,
                                                    train_data, holdout_data)
            results['utility'] = utility_metrics
        
        # Store results
        self.metrics = results
        
        return results