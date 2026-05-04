import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG


class ModelEvaluator:
    """
    Comprehensive model evaluation framework for both regression and classification models
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.benchmark_results = {}
        
    def evaluate_classification_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_proba: Optional[np.ndarray] = None,
                                    model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate classification model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # ROC AUC (if probabilities available)
        roc_auc = None
        if y_proba is not None and len(np.unique(y_true)) == 2:
            roc_auc = roc_auc_score(y_true, y_proba)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def evaluate_regression_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate regression model performance
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Percentage errors
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Direction accuracy (for stock price prediction)
        direction_accuracy = None
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            direction_accuracy = accuracy_score(true_direction, pred_direction)
        
        results = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                           cv_folds: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform cross-validation
        
        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
        
        results = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max(),
            'cv_folds': cv_folds,
            'scoring': scoring
        }
        
        return results
    
    def time_series_cross_validate(self, model, X: np.ndarray, y: np.ndarray,
                                 n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform time series cross-validation
        
        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            n_splits: Number of time series splits
            
        Returns:
            Time series CV results
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
        
        scores = np.array(scores)
        
        results = {
            'tscv_scores': scores.tolist(),
            'tscv_mean': scores.mean(),
            'tscv_std': scores.std(),
            'tscv_min': scores.min(),
            'tscv_max': scores.max(),
            'n_splits': n_splits
        }
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                            model_name: str = "Model") -> go.Figure:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: Class names
            model_name: Model name
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=500,
            height=500
        )
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      model_name: str = "Model") -> go.Figure:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            model_name: Model name
            
        Returns:
            Plotly figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Add diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500
        )
        
        return fig
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str = "Model") -> go.Figure:
        """
        Plot predictions vs actual values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Model name
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', size=6, opacity=0.6)
        ))
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Predictions vs Actual - {model_name}',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            width=600,
            height=500
        )
        
        return fig
    
    def compare_models(self, results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """
        Compare multiple models
        
        Args:
            results: Dictionary of model results
            
        Returns:
            Plotly figure with model comparison
        """
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            values = [results[model].get(metric, 0) for model in models]
            
            fig.add_trace(
                go.Bar(x=models, y=values, name=metric, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Model Comparison',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def generate_evaluation_report(self, results: Dict[str, Any],
                                save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            results: Evaluation results
            save_path: Path to save report
            
        Returns:
            Report content
        """
        report = []
        report.append("# Model Evaluation Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for model_name, metrics in results.items():
            report.append(f"## {model_name}")
            report.append("")
            
            if 'accuracy' in metrics:  # Classification model
                report.append("### Classification Metrics")
                report.append(f"- **Accuracy**: {metrics['accuracy']:.4f}")
                if 'precision' in metrics:
                    report.append(f"- **Precision**: {metrics['precision']:.4f}")
                if 'recall' in metrics:
                    report.append(f"- **Recall**: {metrics['recall']:.4f}")
                if 'f1' in metrics:
                    report.append(f"- **F1 Score**: {metrics['f1']:.4f}")
                if metrics.get('roc_auc'):
                    report.append(f"- **ROC AUC**: {metrics['roc_auc']:.4f}")
                report.append("")
                
            if 'rmse' in metrics:  # Regression model
                report.append("### Regression Metrics")
                report.append(f"- **RMSE**: {metrics['rmse']:.4f}")
                report.append(f"- **MAE**: {metrics['mae']:.4f}")
                report.append(f"- **R²**: {metrics['r2']:.4f}")
                report.append(f"- **MAPE**: {metrics['mape']:.2f}%")
                if metrics.get('direction_accuracy'):
                    report.append(f"- **Direction Accuracy**: {metrics['direction_accuracy']:.4f}")
                report.append("")
            
            if 'cv_mean' in metrics:  # Cross-validation results
                report.append("### Cross-Validation")
                report.append(f"- **CV Mean**: {metrics['cv_mean']:.4f}")
                report.append(f"- **CV Std**: {metrics['cv_std']:.4f}")
                report.append("")
        
        report_content = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
            print(f"Evaluation report saved to {save_path}")
        
        return report_content
    
    def save_evaluation_results(self, results: Dict[str, Any],
                              filepath: str = "evaluation_results.json"):
        """
        Save evaluation results to JSON file
        
        Args:
            results: Evaluation results
            filepath: Path to save results
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Evaluation results saved to {filepath}")


def evaluate_all_models() -> Dict[str, Any]:
    """
    Evaluate all available models in the project
    
    Returns:
        Comprehensive evaluation results
    """
    evaluator = ModelEvaluator()
    results = {}
    
    # Check for classification models
    models_dir = 'models'
    if os.path.exists(models_dir):
        # Check for classification results
        classification_results_path = os.path.join(models_dir, 'classification_results.json')
        if os.path.exists(classification_results_path):
            with open(classification_results_path, 'r') as f:
                classification_data = json.load(f)
            
            for model_name, metrics in classification_data['results'].items():
                results[f"{model_name}_classification"] = metrics
        
        # Check for Naive Bayes metrics
        nb_metrics_path = os.path.join(models_dir, 'naive_bayes_metrics.json')
        if os.path.exists(nb_metrics_path):
            with open(nb_metrics_path, 'r') as f:
                nb_metrics = json.load(f)
            results['naive_bayes_sentiment'] = nb_metrics
    
    # Check for ARIMA results
    arima_summary_path = os.path.join(DATA_CONFIG['processed_data_dir'], 'arima_summary.json')
    if os.path.exists(arima_summary_path):
        with open(arima_summary_path, 'r') as f:
            arima_data = json.load(f)
        
        # Convert ARIMA results to evaluation format
        for symbol_data in arima_data:
            symbol = symbol_data.get('symbol', 'Unknown')
            results[f"arima_{symbol}"] = {
                'model_name': f"ARIMA_{symbol}",
                'rmse': symbol_data.get('rmse', 0),
                'mae': symbol_data.get('mae', 0),
                'r2': symbol_data.get('r2', 0),
                'mape': symbol_data.get('mape', 0)
            }
    
    # Generate report
    if results:
        report = evaluator.generate_evaluation_report(results)
        evaluator.save_evaluation_results(results)
        
        print("Model Evaluation Summary:")
        print("=" * 50)
        for model_name, metrics in results.items():
            if 'accuracy' in metrics:
                f1_str = f", F1={metrics['f1']:.4f}" if 'f1' in metrics else ""
                print(f"{model_name}: Accuracy={metrics['accuracy']:.4f}{f1_str}")
            elif 'rmse' in metrics:
                print(f"{model_name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    else:
        print("No model results found for evaluation")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate all models")
    parser.add_argument("--output_dir", default="evaluation",
                       help="Directory to save evaluation results")
    args = parser.parse_args()

    print("Evaluating all models...")
    results = evaluate_all_models()
    
    if results:
        # Save results to specified directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        evaluator = ModelEvaluator()
        evaluator.save_evaluation_results(results, os.path.join(args.output_dir, 'all_models_evaluation.json'))
        
        report_path = os.path.join(args.output_dir, 'evaluation_report.md')
        evaluator.generate_evaluation_report(results, report_path)
        
        print(f"\nEvaluation complete! Results saved to {args.output_dir}/")
