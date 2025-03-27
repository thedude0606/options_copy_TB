"""
Ensemble Models for Options Trading Prediction.
Implements various ensemble learning approaches for improved prediction accuracy.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import os
import datetime as dt
import json

class StackedEnsembleModel(BaseEstimator, RegressorMixin):
    """
    Stacked ensemble model that combines multiple base models with a meta-learner.
    """
    def __init__(self, base_models=None, meta_model=None, cv=5, use_features_in_meta=False):
        """
        Initialize the stacked ensemble model.
        
        Parameters:
        -----------
        base_models : list
            List of base models (estimators)
        meta_model : estimator
            Meta-learner model
        cv : int
            Number of cross-validation folds for training
        use_features_in_meta : bool
            Whether to include original features in meta-learner input
        """
        self.base_models = base_models or self._default_base_models()
        self.meta_model = meta_model or LinearRegression()
        self.cv = cv
        self.use_features_in_meta = use_features_in_meta
        self.base_models_fitted = False
        self.meta_model_fitted = False
        self.feature_names = None
        
    def _default_base_models(self):
        """
        Create default base models for the ensemble.
        
        Returns:
        --------
        list
            List of default base models
        """
        models = [
            RandomForestRegressor(n_estimators=100, random_state=42),
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            xgb.XGBRegressor(n_estimators=100, random_state=42),
            lgb.LGBMRegressor(n_estimators=100, random_state=42),
            MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        ]
        return models
    
    def fit(self, X, y):
        """
        Fit the stacked ensemble model.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Store feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Convert to numpy arrays if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Create empty array to store meta-features
        meta_features = np.zeros((X_array.shape[0], len(self.base_models)))
        
        # Train base models using cross-validation
        for i, model in enumerate(self.base_models):
            # Use cross-validation to create out-of-fold predictions
            for train_idx, val_idx in self._get_cv_splits(X_array):
                # Split data
                X_train, X_val = X_array[train_idx], X_array[val_idx]
                y_train, y_val = y_array[train_idx], y_array[val_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Generate predictions for validation fold
                meta_features[val_idx, i] = model.predict(X_val)
            
            # Retrain model on full dataset
            model.fit(X_array, y_array)
        
        # Mark base models as fitted
        self.base_models_fitted = True
        
        # Prepare meta-features for meta-model training
        if self.use_features_in_meta:
            # Combine meta-features with original features
            meta_input = np.hstack((meta_features, X_array))
        else:
            meta_input = meta_features
        
        # Train meta-model
        self.meta_model.fit(meta_input, y_array)
        
        # Mark meta-model as fitted
        self.meta_model_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Generate predictions using the stacked ensemble model.
        
        Parameters:
        -----------
        X : array-like
            Input data
            
        Returns:
        --------
        array-like
            Predicted values
        """
        # Check if models are fitted
        if not self.base_models_fitted or not self.meta_model_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Convert to numpy array if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Generate predictions from base models
        meta_features = np.column_stack([
            model.predict(X_array) for model in self.base_models
        ])
        
        # Prepare meta-features for meta-model prediction
        if self.use_features_in_meta:
            # Combine meta-features with original features
            meta_input = np.hstack((meta_features, X_array))
        else:
            meta_input = meta_features
        
        # Generate final predictions using meta-model
        return self.meta_model.predict(meta_input)
    
    def _get_cv_splits(self, X):
        """
        Generate cross-validation splits.
        
        Parameters:
        -----------
        X : array-like
            Input data
            
        Returns:
        --------
        generator
            Generator of train and validation indices
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        fold_size = n_samples // self.cv
        
        for i in range(self.cv):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < self.cv - 1 else n_samples
            
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
            
            yield train_indices, val_indices
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters:
        -----------
        deep : bool
            If True, return the parameters of all sub-objects
            
        Returns:
        --------
        dict
            Parameter names mapped to their values
        """
        return {
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'cv': self.cv,
            'use_features_in_meta': self.use_features_in_meta
        }
    
    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        
        Parameters:
        -----------
        **parameters : dict
            Estimator parameters
            
        Returns:
        --------
        self : object
            Returns self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def feature_importances(self):
        """
        Get feature importances from the ensemble model.
        
        Returns:
        --------
        dict
            Feature importance scores for each model and combined
        """
        if not self.base_models_fitted:
            raise RuntimeError("Model must be fitted before getting feature importances")
        
        importances = {}
        
        # Get feature importances from base models
        for i, model in enumerate(self.base_models):
            model_name = type(model).__name__
            
            if hasattr(model, 'feature_importances_'):
                model_importances = model.feature_importances_
                
                # Map to feature names if available
                if self.feature_names is not None and len(self.feature_names) == len(model_importances):
                    model_importances = dict(zip(self.feature_names, model_importances))
                
                importances[model_name] = model_importances
        
        # Calculate combined importance (weighted average)
        if importances:
            # Extract all feature names
            all_features = set()
            for model_importances in importances.values():
                if isinstance(model_importances, dict):
                    all_features.update(model_importances.keys())
            
            # Initialize combined importances
            combined = {feature: 0.0 for feature in all_features}
            
            # Sum importances across models
            for model_importances in importances.values():
                if isinstance(model_importances, dict):
                    for feature, importance in model_importances.items():
                        combined[feature] += importance
            
            # Normalize
            total = sum(combined.values())
            if total > 0:
                combined = {feature: importance / total for feature, importance in combined.items()}
            
            importances['combined'] = combined
        
        return importances


class WeightedEnsembleModel(BaseEstimator, RegressorMixin):
    """
    Weighted ensemble model that combines predictions from multiple models.
    """
    def __init__(self, models=None, weights=None, optimize_weights=True):
        """
        Initialize the weighted ensemble model.
        
        Parameters:
        -----------
        models : list
            List of models (estimators)
        weights : array-like
            Weights for each model
        optimize_weights : bool
            Whether to optimize weights during fitting
        """
        self.models = models or self._default_models()
        self.weights = weights
        self.optimize_weights = optimize_weights
        self.models_fitted = False
        
    def _default_models(self):
        """
        Create default models for the ensemble.
        
        Returns:
        --------
        list
            List of default models
        """
        models = [
            RandomForestRegressor(n_estimators=100, random_state=42),
            xgb.XGBRegressor(n_estimators=100, random_state=42),
            lgb.LGBMRegressor(n_estimators=100, random_state=42),
            cb.CatBoostRegressor(n_estimators=100, random_state=42, verbose=0)
        ]
        return models
    
    def fit(self, X, y):
        """
        Fit the weighted ensemble model.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Split data for weight optimization
        if self.optimize_weights:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, y_train = X, y
        
        # Train each model
        for model in self.models:
            model.fit(X_train, y_train)
        
        # Mark models as fitted
        self.models_fitted = True
        
        # Optimize weights if enabled
        if self.optimize_weights:
            self._optimize_weights(X_val, y_val)
        elif self.weights is None:
            # Use equal weights if not optimizing and weights not provided
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        return self
    
    def predict(self, X):
        """
        Generate predictions using the weighted ensemble model.
        
        Parameters:
        -----------
        X : array-like
            Input data
            
        Returns:
        --------
        array-like
            Predicted values
        """
        # Check if models are fitted
        if not self.models_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Generate predictions from each model
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        
        # Apply weights to predictions
        return np.sum(predictions * self.weights.reshape(-1, 1), axis=0)
    
    def _optimize_weights(self, X_val, y_val):
        """
        Optimize model weights using validation data.
        
        Parameters:
        -----------
        X_val : array-like
            Validation data
        y_val : array-like
            Validation target values
        """
        # Generate predictions for validation data
        val_predictions = np.column_stack([
            model.predict(X_val) for model in self.models
        ])
        
        # Calculate errors for each model
        errors = np.abs(val_predictions - y_val.reshape(-1, 1))
        mean_errors = np.mean(errors, axis=0)
        
        # Convert errors to weights (lower error = higher weight)
        if np.all(mean_errors == 0):
            # If all errors are zero, use equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)
        else:
            # Invert errors and normalize
            weights = 1.0 / (mean_errors + 1e-10)
            self.weights = weights / np.sum(weights)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters:
        -----------
        deep : bool
            If True, return the parameters of all sub-objects
            
        Returns:
        --------
        dict
            Parameter names mapped to their values
        """
        return {
            'models': self.models,
            'weights': self.weights,
            'optimize_weights': self.optimize_weights
        }
    
    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        
        Parameters:
        -----------
        **parameters : dict
            Estimator parameters
            
        Returns:
        --------
        self : object
            Returns self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class BoostedEnsembleModel(BaseEstimator, RegressorMixin):
    """
    Boosted ensemble model that sequentially trains models on residuals.
    """
    def __init__(self, base_models=None, learning_rate=0.1, n_estimators=3):
        """
        Initialize the boosted ensemble model.
        
        Parameters:
        -----------
        base_models : list
            List of base model classes to use in boosting
        learning_rate : float
            Learning rate for boosting
        n_estimators : int
            Number of estimators to train
        """
        self.base_models = base_models or self._default_base_models()
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.models = []
        self.models_fitted = False
        
    def _default_base_models(self):
        """
        Create default base model classes for the ensemble.
        
        Returns:
        --------
        list
            List of default base model classes
        """
        return [
            xgb.XGBRegressor,
            lgb.LGBMRegressor,
            RandomForestRegressor
        ]
    
    def fit(self, X, y):
        """
        Fit the boosted ensemble model.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Convert to numpy arrays if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Initialize models list
        self.models = []
        
        # Initialize residuals with target values
        residuals = y_array.copy()
        
        # Train models sequentially
        for i in range(self.n_estimators):
            # Select base model class (cycling through available models)
            model_class = self.base_models[i % len(self.base_models)]
            
            # Create model instance with random state for reproducibility
            model = model_class(random_state=42)
            
            # Train model on current residuals
            model.fit(X_array, residuals)
            
            # Add model to ensemble
            self.models.append(model)
            
            # Update residuals
            predictions = model.predict(X_array)
            residuals -= self.learning_rate * predictions
        
        # Mark models as fitted
        self.models_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Generate predictions using the boosted ensemble model.
        
        Parameters:
        -----------
        X : array-like
            Input data
            
        Returns:
        --------
        array-like
            Predicted values
        """
        # Check if models are fitted
        if not self.models_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Convert to numpy array if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Initialize predictions with zeros
        predictions = np.zeros(X_array.shape[0])
        
        # Add predictions from each model
        for model in self.models:
            predictions += self.learning_rate * model.predict(X_array)
        
        return predictions
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters:
        -----------
        deep : bool
            If True, return the parameters of all sub-objects
            
        Returns:
        --------
        dict
            Parameter names mapped to their values
        """
        return {
            'base_models': self.base_models,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators
        }
    
    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        
        Parameters:
        -----------
        **parameters : dict
            Estimator parameters
            
        Returns:
        --------
        self : object
            Returns self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class ModelManager:
    """
    Manager class for ensemble models.
    Handles model training, evaluation, and persistence.
    """
    def __init__(self, model_dir='models'):
        """
        Initialize the model manager.
        
        Parameters:
        -----------
        model_dir : str
            Directory to store trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.metrics = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def train_model(self, model_type, X, y, model_name=None, **kwargs):
        """
        Train a model and store it.
        
        Parameters:
        -----------
        model_type : str
            Type of model to train ('stacked', 'weighted', or 'boosted')
        X : array-like
            Training data
        y : array-like
            Target values
        model_name : str, optional
            Name for the model (generated if not provided)
        **kwargs : dict
            Additional parameters for the model
            
        Returns:
        --------
        str
            Name of the trained model
        """
        # Generate model name if not provided
        if model_name is None:
            timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{model_type}_{timestamp}"
        
        # Create model based on type
        if model_type == 'stacked':
            model = StackedEnsembleModel(**kwargs)
        elif model_type == 'weighted':
            model = WeightedEnsembleModel(**kwargs)
        elif model_type == 'boosted':
            model = BoostedEnsembleModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X, y)
        
        # Store model
        self.models[model_name] = model
        
        # Evaluate model
        self.evaluate_model(model_name, X, y)
        
        return model_name
    
    def evaluate_model(self, model_name, X, y):
        """
        Evaluate a model and store metrics.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to evaluate
        X : array-like
            Evaluation data
        y : array-like
            Target values
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        # Get model
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model not found: {model_name}")
        
        # Generate predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        # Store metrics
        self.metrics[model_name] = metrics
        
        return metrics
    
    def save_model(self, model_name):
        """
        Save a model to disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to save
            
        Returns:
        --------
        str
            Path to the saved model
        """
        # Get model
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model not found: {model_name}")
        
        # Create model path
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save metrics if available
        metrics = self.metrics.get(model_name)
        if metrics is not None:
            metrics_path = os.path.join(self.model_dir, f"{model_name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
        
        return model_path
    
    def load_model(self, model_name):
        """
        Load a model from disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to load
            
        Returns:
        --------
        object
            Loaded model
        """
        # Create model path
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Store model
        self.models[model_name] = model
        
        # Load metrics if available
        metrics_path = os.path.join(self.model_dir, f"{model_name}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics[model_name] = json.load(f)
        
        return model
    
    def get_best_model(self, metric='rmse'):
        """
        Get the best model based on a metric.
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison
            
        Returns:
        --------
        tuple
            (model_name, model) of the best model
        """
        if not self.metrics:
            raise ValueError("No models have been evaluated")
        
        # Find best model based on metric
        if metric in ['mse', 'rmse', 'mae']:
            # Lower is better
            best_model_name = min(self.metrics, key=lambda k: self.metrics[k].get(metric, float('inf')))
        else:
            # Higher is better
            best_model_name = max(self.metrics, key=lambda k: self.metrics[k].get(metric, float('-inf')))
        
        return best_model_name, self.models[best_model_name]
    
    def list_models(self):
        """
        List all available models with their metrics.
        
        Returns:
        --------
        dict
            Dictionary of model names and their metrics
        """
        return {name: self.metrics.get(name, {}) for name in self.models}
    
    def delete_model(self, model_name):
        """
        Delete a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to delete
        """
        # Remove from memory
        if model_name in self.models:
            del self.models[model_name]
        
        if model_name in self.metrics:
            del self.metrics[model_name]
        
        # Remove from disk
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        if os.path.exists(model_path):
            os.remove(model_path)
        
        metrics_path = os.path.join(self.model_dir, f"{model_name}_metrics.json")
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
