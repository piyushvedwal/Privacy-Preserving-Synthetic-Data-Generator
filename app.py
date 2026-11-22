# app.py
# Streamlit interface for Privacy-Preserving Synthetic Data Generator

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any

# Import custom modules
from data_preprocess import DataPreprocessor
from model_train import PrivacyPreservingCTGAN
from evaluate import SyntheticDataEvaluator
from privacy_checks import distance_to_closest_record
from sample_datasets import SAMPLE_DATASETS, load_sample
from report import build_report_data, build_markdown_report, build_json_report

# Set page configuration
st.set_page_config(
    page_title="Privacy-Preserving Synthetic Data Generator",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'real_data' not in st.session_state:
    st.session_state.real_data = None
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'column_types' not in st.session_state:
    st.session_state.column_types = None
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'dcr_results' not in st.session_state:
    st.session_state.dcr_results = None
if 'run_meta' not in st.session_state:
    st.session_state.run_meta = {}
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None

def reset_downstream_state() -> None:
    """Clear model, synthetic data and evaluation when a new dataset is loaded."""
    for key in ['model', 'training_metrics', 'synthetic_data', 'evaluation_results', 'dcr_results']:
        st.session_state[key] = None
    st.session_state.run_meta = {}

# Helper functions
def save_dataframe(df: pd.DataFrame, filename: str) -> str:
    """Save DataFrame to CSV and return the path."""
    # Create 'output' directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"output/{timestamp}_{filename}"
    
    # Save to CSV
    df.to_csv(path, index=False)
    return path

def display_metrics(metrics: Dict[str, Any]) -> None:
    """Display evaluation metrics in a formatted way."""
    if not metrics:
        st.warning("No metrics available. Please run evaluation first.")
        return
    
    # Fidelity metrics
    if 'fidelity' in metrics:
        st.subheader("Fidelity Metrics")
        fidelity = metrics['fidelity']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'overall_fidelity' in fidelity and fidelity['overall_fidelity'] is not None:
                st.metric("Overall Fidelity", f"{fidelity['overall_fidelity']:.4f}")
            
            if 'ks_mean' in fidelity and fidelity['ks_mean'] is not None:
                st.metric("KS Similarity (Mean)", f"{fidelity['ks_mean']:.4f}")
        
        with col2:
            if 'correlation_similarity' in fidelity and fidelity['correlation_similarity'] is not None:
                st.metric("Correlation Similarity", f"{fidelity['correlation_similarity']:.4f}")
            
            if 'chi2_mean' in fidelity and fidelity['chi2_mean'] is not None:
                st.metric("Chi-Square Similarity (Mean)", f"{fidelity['chi2_mean']:.4f}")
        
        with col3:
            if 'pmse_similarity' in fidelity and fidelity['pmse_similarity'] is not None:
                st.metric("Propensity Similarity", f"{fidelity['pmse_similarity']:.4f}")
            
            if 'real_vs_synthetic_auc' in fidelity and fidelity['real_vs_synthetic_auc'] is not None:
                st.metric("Real vs Synthetic AUC", f"{fidelity['real_vs_synthetic_auc']:.4f}")
    
    # Privacy metrics
    if 'privacy' in metrics:
        st.subheader("Privacy Metrics")
        privacy = metrics['privacy']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'privacy_score' in privacy and privacy['privacy_score'] is not None:
                st.metric("Privacy Score", f"{privacy['privacy_score']:.4f}")
        
        with col2:
            if 'membership_inference_auc' in privacy and privacy['membership_inference_auc'] is not None:
                st.metric("Membership Inference AUC", f"{privacy['membership_inference_auc']:.4f}")
        
        with col3:
            if 'epsilon' in privacy and privacy['epsilon'] is not None:
                st.metric("Differential Privacy ε", f"{privacy['epsilon']:.2f}")
                if 'delta' in privacy and privacy['delta'] is not None:
                    st.metric("Differential Privacy δ", f"{privacy['delta']:.1e}")
    
    # Utility metrics
    if 'utility' in metrics:
        st.subheader("Utility Metrics")
        utility = metrics['utility']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'utility_ratio' in utility and utility['utility_ratio'] is not None:
                st.metric("Utility Ratio", f"{utility['utility_ratio']:.4f}")
        
        with col2:
            if 'real_model_accuracy' in utility and utility['real_model_accuracy'] is not None:
                st.metric("Real Model Accuracy", f"{utility['real_model_accuracy']:.4f}")
        
        with col3:
            if 'synthetic_model_accuracy' in utility and utility['synthetic_model_accuracy'] is not None:
                st.metric("Synthetic Model Accuracy", f"{utility['synthetic_model_accuracy']:.4f}")

def display_dcr(dcr: Dict[str, Any]) -> None:
    """Show the Distance to Closest Record privacy check."""
    st.subheader("Record-Level Privacy (Distance to Closest Record)")
    st.caption(
        "Checks whether synthetic rows copy or sit unusually close to real rows. "
        "Each synthetic row's distance to its nearest real row is compared with how far "
        "real rows sit from each other."
    )
    
    risk = dcr['risk_level']
    message = {
        "Low": "✅ Low re-identification risk: synthetic rows are not unusually close to real records.",
        "Medium": "⚠️ Medium risk: some synthetic rows sit closer to real records than expected.",
        "High": "🚨 High risk: synthetic rows copy or nearly copy real records. Try a lower ε or fewer epochs.",
    }[risk]
    {"Low": st.success, "Medium": st.warning, "High": st.error}[risk](message)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ratio = dcr['dcr_ratio']
        st.metric("DCR Ratio (synthetic / baseline)", f"{ratio:.3f}" if ratio is not None else "n/a")
    with col2:
        st.metric("Exact Copies of Real Rows", f"{dcr['exact_match_count']} ({dcr['exact_match_rate']:.2%})")
    with col3:
        st.metric("Median DCR (synthetic → real)", f"{dcr['synthetic_dcr_median']:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    upper = np.percentile(np.concatenate([dcr['synthetic_dcr_values'], dcr['real_dcr_values']]), 99)
    bins = np.linspace(0, upper if upper > 0 else 1, 40)
    ax.hist(dcr['real_dcr_values'], bins=bins, alpha=0.6, label="Real → nearest other real (baseline)")
    ax.hist(dcr['synthetic_dcr_values'], bins=bins, alpha=0.6, label="Synthetic → nearest real")
    ax.set_xlabel("Distance to closest record")
    ax.set_ylabel("Rows")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Main application
st.title("🔒 Privacy-Preserving Synthetic Data Generator")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a step:",
    ["1. Upload Dataset", "2. Train Model", "3. Generate Synthetic Data", "4. Evaluate & Visualize"]
)

# 1. Upload Dataset
if page == "1. Upload Dataset":
    st.header("Upload Dataset")
    
    upload_method = st.radio(
        "Choose upload method:",
        ["Use a sample dataset", "Upload CSV file", "Enter file path"]
    )
    
    data_loaded = False
    dataset_name = None
    
    if upload_method == "Use a sample dataset":
        st.caption("No CSV handy? Try the app with a built-in dataset.")
        sample_name = st.selectbox("Choose a sample dataset:", list(SAMPLE_DATASETS.keys()))
        if st.button("Load Sample Dataset"):
            try:
                df = load_sample(sample_name)
                data_loaded = True
                dataset_name = sample_name
            except Exception as e:
                st.error(f"Error loading sample dataset: {e}")
    elif upload_method == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                data_loaded = True
                dataset_name = uploaded_file.name
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:  # Enter file path
        file_path = st.text_input("Enter the path to a CSV file:")
        
        if file_path and st.button("Load Data"):
            try:
                df = pd.read_csv(file_path)
                data_loaded = True
                dataset_name = os.path.basename(file_path)
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    if data_loaded:
        new_dataset = (
            st.session_state.real_data is None
            or dataset_name != st.session_state.dataset_name
            or not st.session_state.real_data.equals(df)
        )
        if new_dataset:
            reset_downstream_state()
        st.session_state.real_data = df
        st.session_state.dataset_name = dataset_name
        st.success(f"Dataset loaded successfully! Shape: {df.shape}")
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Display data info
        st.subheader("Data Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Statistics:**")
            st.write(f"- Rows: {df.shape[0]}")
            st.write(f"- Columns: {df.shape[1]}")
            st.write(f"- Missing values: {df.isna().sum().sum()}")
        
        with col2:
            st.write("**Column Types:**")
            dtypes = df.dtypes.value_counts().to_dict()
            for dtype, count in dtypes.items():
                st.write(f"- {dtype}: {count} columns")
        
        # Initialize preprocessor and detect column types
        preprocessor = DataPreprocessor()
        column_types = preprocessor.detect_column_types(df)
        
        # Store in session state
        st.session_state.preprocessor = preprocessor
        st.session_state.column_types = column_types
        
        # Display detected column types
        st.subheader("Detected Column Types")
        col_type_df = pd.DataFrame({
            'Column': list(column_types.keys()),
            'Type': list(column_types.values())
        })
        st.dataframe(col_type_df)
        
        st.info("✅ Dataset loaded and preprocessed. Proceed to the 'Train Model' step.")

# 2. Train Model
elif page == "2. Train Model":
    st.header("Train Synthetic Data Model")
    
    if st.session_state.real_data is None:
        st.warning("Please upload a dataset first.")
    else:
        # Model configuration
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_dp = st.checkbox("Use Differential Privacy", value=True)
            epochs = st.slider("Training Epochs", min_value=50, max_value=500,
                               value=100 if use_dp else 300, step=50,
                               help="With differential privacy every extra epoch costs privacy budget, "
                                    "so fewer epochs (around 100) usually give better data.")
            batch_size = st.slider("Batch Size", min_value=100, max_value=1000, value=500, step=100)
        
        with col2:
            if use_dp:
                epsilon = st.slider("Privacy Budget (ε)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                delta = st.number_input("Privacy Failure Probability (δ)", 
                                       min_value=1e-7, max_value=1e-3, value=1e-5, format="%e")
                max_grad_norm = st.slider("Max Gradient Norm", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            else:
                epsilon = 0.0
                delta = 0.0
                max_grad_norm = 1.0
        
        if use_dp:
            n_rows = len(st.session_state.real_data)
            st.info(
                "**How the privacy guarantee works:** a GAN is trained where only the discriminator sees "
                "real records, using DP-SGD (per-record gradient clipping plus calibrated Gaussian noise, via "
                "Opacus). The generator never sees real data, so every synthetic row inherits the "
                "(ε, δ) guarantee. Column statistics used for scaling and the list of categories are "
                "taken from the data without noise."
            )
            if n_rows < 1000 or epsilon < 5:
                st.warning(
                    f"Your dataset has {n_rows} rows and ε = {epsilon}. Differential privacy adds a lot of noise "
                    "on small datasets or small ε, so expect lower fidelity. Larger datasets (thousands of rows) "
                    "or ε around 5–10 give noticeably better synthetic data."
                )
            if delta >= 1.0 / n_rows:
                st.warning(f"δ should be smaller than 1 / number of rows (1/{n_rows} = {1.0 / n_rows:.1e}).")
        
        # Train button
        if st.button("Train Model"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Get data
                    df = st.session_state.real_data
                    preprocessor = st.session_state.preprocessor
                    column_types = st.session_state.column_types
                    
                    # Preprocess data
                    processed_df = preprocessor.fit_transform(df, column_types)
                    
                    # Get discrete columns for CTGAN
                    discrete_columns = [col for col in processed_df.columns 
                                       if any(col.startswith(f"{c}_") for c in preprocessor.categorical_columns)]
                    
                    # Initialize and train model
                    model = PrivacyPreservingCTGAN(
                        use_dp=use_dp,
                        epsilon=epsilon,
                        delta=delta,
                        max_grad_norm=max_grad_norm,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=True
                    )
                    
                    # Train model
                    training_metrics = model.train(
                        processed_df, discrete_columns, preprocessor.one_hot_encoding_maps
                    )
                    
                    # Store model and metrics in session state
                    st.session_state.model = model
                    st.session_state.training_metrics = training_metrics
                    st.session_state.synthetic_data = None
                    st.session_state.evaluation_results = None
                    st.session_state.dcr_results = None
                    st.session_state.run_meta = {
                        "Dataset": st.session_state.dataset_name,
                        "Real rows": len(df),
                        "Columns": df.shape[1],
                        "Differential privacy": "On" if use_dp else "Off",
                        "Target ε": epsilon if use_dp else None,
                        "δ": delta if use_dp else None,
                        "Epochs": epochs,
                        "Batch size": batch_size,
                        "Training time (s)": round(training_metrics['training_time'], 2),
                    }
                    if use_dp:
                        st.session_state.run_meta.update({
                            "ε spent": round(training_metrics['epsilon'], 4),
                            "Noise multiplier": round(training_metrics['noise_multiplier'], 3),
                            "Max gradient norm": max_grad_norm,
                            "Sampling rate": round(training_metrics['sample_rate'], 4),
                            "DP-SGD steps": training_metrics['steps'],
                        })
                    
                    st.success("Model trained successfully!")
                    
                    # Display training metrics
                    st.subheader("Training Metrics")
                    st.write(f"Training time: {training_metrics['training_time']:.2f} seconds")
                    st.write(f"Epochs: {training_metrics['epochs']}")
                    
                    if use_dp:
                        st.write(f"Final privacy guarantee: (ε={training_metrics['epsilon']:.2f}, δ={training_metrics['delta']:.1e})")
                        st.write(
                            f"DP-SGD: noise multiplier {training_metrics['noise_multiplier']:.2f}, "
                            f"sampling rate {training_metrics['sample_rate']:.3f} "
                            f"(expected batch {training_metrics['expected_batch_size']}), "
                            f"{training_metrics['steps']} steps"
                        )
                    
                    st.info("✅ Model trained successfully. Proceed to 'Generate Synthetic Data' step.")
                except Exception as e:
                    st.error(f"Error during training: {e}")

# 3. Generate Synthetic Data
elif page == "3. Generate Synthetic Data":
    st.header("Generate Synthetic Data")
    
    if st.session_state.model is None:
        st.warning("Please train a model first.")
    else:
        # Generation configuration
        st.subheader("Generation Configuration")
        
        # Number of rows to generate
        num_rows = st.slider(
            "Number of synthetic rows to generate", 
            min_value=100, 
            max_value=10000, 
            value=min(5000, len(st.session_state.real_data)),
            step=100
        )
        
        # Generate button
        if st.button("Generate Synthetic Data"):
            with st.spinner("Generating synthetic data..."):
                try:
                    # Get model and preprocessor
                    model = st.session_state.model
                    preprocessor = st.session_state.preprocessor
                    
                    # Generate synthetic data
                    synthetic_processed = model.generate(num_rows)
                    
                    # Inverse transform to original format
                    synthetic_data = preprocessor.inverse_transform(synthetic_processed)
                    
                    # Store in session state
                    st.session_state.synthetic_data = synthetic_data
                    
                    st.success(f"Generated {len(synthetic_data)} synthetic rows successfully!")
                    
                    # Save synthetic data
                    output_path = save_dataframe(synthetic_data, "synthetic_data.csv")
                    st.info(f"Synthetic data saved to: {output_path}")
                    
                    st.session_state.evaluation_results = None
                    st.session_state.dcr_results = None
                    st.session_state.run_meta["Synthetic rows"] = len(synthetic_data)
                    
                    st.info("✅ Synthetic data generated. Proceed to 'Evaluate & Visualize' step.")
                except Exception as e:
                    st.error(f"Error generating synthetic data: {e}")
        
        # Download and preview stay visible across reruns
        if st.session_state.synthetic_data is not None:
            synthetic_data = st.session_state.synthetic_data
            st.download_button(
                label="Download Synthetic Data",
                data=synthetic_data.to_csv(index=False).encode("utf-8"),
                file_name="synthetic_data.csv",
                mime="text/csv"
            )
            st.subheader("Synthetic Data Preview")
            st.dataframe(synthetic_data.head())

# 4. Evaluate & Visualize
elif page == "4. Evaluate & Visualize":
    st.header("Evaluate & Visualize")
    
    if st.session_state.real_data is None or st.session_state.synthetic_data is None:
        st.warning("Please generate synthetic data first.")
    else:
        # Get data
        real_data = st.session_state.real_data
        synthetic_data = st.session_state.synthetic_data
        
        # Evaluation options
        st.subheader("Evaluation Options")
        
        # Target column for utility evaluation
        target_column = None
        if len(real_data.columns) > 0:
            target_column = st.selectbox(
                "Select target column for utility evaluation (optional):",
                [None] + list(real_data.columns)
            )
        
        # Evaluate button
        if st.button("Run Evaluation"):
            with st.spinner("Running evaluation..."):
                try:
                    # Initialize evaluator
                    evaluator = SyntheticDataEvaluator()
                    
                    # Set column types if available
                    if st.session_state.column_types is not None:
                        evaluator.set_column_types(st.session_state.column_types)
                    
                    # Get privacy metrics if available
                    privacy_metrics = None
                    if st.session_state.training_metrics is not None and 'epsilon' in st.session_state.training_metrics:
                        privacy_metrics = {
                            'epsilon': st.session_state.training_metrics['epsilon'],
                            'delta': st.session_state.training_metrics['delta']
                        }
                    
                    # Run evaluation
                    results = evaluator.evaluate_all(
                        real_data=real_data,
                        synthetic_data=synthetic_data,
                        privacy_metrics=privacy_metrics,
                        target_column=target_column
                    )
                    
                    # Store results
                    st.session_state.evaluation_results = results
                    
                    # Record-level privacy check (Distance to Closest Record)
                    try:
                        st.session_state.dcr_results = distance_to_closest_record(real_data, synthetic_data)
                    except Exception as dcr_error:
                        st.session_state.dcr_results = None
                        st.warning(f"Could not run the Distance to Closest Record check: {dcr_error}")
                    
                    st.success("Evaluation completed successfully!")
                    
                    # Save evaluation results
                    os.makedirs('output', exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_path = f"output/{timestamp}_evaluation_results.json"
                    
                    # Convert numpy values to Python native types for JSON serialization
                    def convert_to_serializable(obj):
                        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                                           np.uint8, np.uint16, np.uint32, np.uint64)):
                            return int(obj)
                        elif isinstance(obj, (np.float64, np.float16, np.float32)):
                            return float(obj)
                        elif isinstance(obj, (np.ndarray,)):
                            return obj.tolist()
                        elif isinstance(obj, dict):
                            return {k: convert_to_serializable(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_to_serializable(i) for i in obj]
                        else:
                            return obj
                    
                    serializable_results = convert_to_serializable(results)
                    
                    with open(results_path, 'w') as f:
                        json.dump(serializable_results, f, indent=2)
                    
                    st.info(f"Evaluation results saved to: {results_path}")
                except Exception as e:
                    st.error(f"Error during evaluation: {e}")
        
        # Display evaluation results if available
        if st.session_state.evaluation_results is not None:
            display_metrics(st.session_state.evaluation_results)
            
            dcr = st.session_state.dcr_results
            if dcr is not None:
                display_dcr(dcr)
            
            # Downloadable report
            st.subheader("Download Report")
            report_data = build_report_data(
                st.session_state.evaluation_results,
                dcr,
                st.session_state.run_meta,
            )
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Report (Markdown)",
                    data=build_markdown_report(report_data).encode("utf-8"),
                    file_name="synthetic_data_report.md",
                    mime="text/markdown",
                )
            with col2:
                st.download_button(
                    label="Download Report (JSON)",
                    data=build_json_report(report_data).encode("utf-8"),
                    file_name="synthetic_data_report.json",
                    mime="application/json",
                )
            
            # Visualizations
            st.subheader("Visualizations")
            
            # Column selection for distribution plots
            if len(real_data.columns) > 0:
                # Get numerical columns
                numerical_cols = [col for col in real_data.columns 
                                if pd.api.types.is_numeric_dtype(real_data[col])]
                
                if len(numerical_cols) > 0:
                    selected_columns = st.multiselect(
                        "Select columns for distribution plots:",
                        options=real_data.columns,
                        default=numerical_cols[:min(5, len(numerical_cols))]
                    )
                    
                    if selected_columns:
                        # Create distribution plots
                        evaluator = SyntheticDataEvaluator()
                        fig = evaluator.plot_distributions(
                            real_data=real_data,
                            synthetic_data=synthetic_data,
                            columns=selected_columns,
                            figsize=(15, 10)
                        )
                        st.pyplot(fig)
                
                # Correlation heatmap
                if st.checkbox("Show Correlation Heatmaps", value=True):
                    evaluator = SyntheticDataEvaluator()
                    fig = evaluator.plot_correlation_comparison(
                        real_data=real_data,
                        synthetic_data=synthetic_data,
                        figsize=(15, 7)
                    )
                    st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "Privacy-Preserving Synthetic Data Generator using CTGAN and Differential Privacy. "
    "This tool allows you to generate synthetic data that preserves statistical patterns "
    "while protecting sensitive information."
)

# Display current state
if st.sidebar.checkbox("Show Debug Info", value=False):
    st.sidebar.subheader("Current State")
    state_info = {
        "Data Loaded": st.session_state.real_data is not None,
        "Model Trained": st.session_state.model is not None,
        "Synthetic Data Generated": st.session_state.synthetic_data is not None,
        "Evaluation Complete": st.session_state.evaluation_results is not None,
        "DCR Check Complete": st.session_state.dcr_results is not None
    }
    for key, value in state_info.items():
        st.sidebar.write(f"{key}: {value}")