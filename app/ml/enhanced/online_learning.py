"""
Online Learning Components for Options Trading Models.
Implements adaptive learning capabilities for continuous model updates.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from river import linear_model, preprocessing, compose, metrics, drift
import joblib
import os
import datetime as dt
import json
import pickle

class OnlineEnsembleModel(BaseEstimator, RegressorMixin):
    """
    Online ensemble model that continuously updates with new data.
    Uses River library for incremental learning.
    """
    def __init__(self, models=None, weights=None, drift_detector=True, 
                 feature_scaler=True, learning_rate=0.01):
        """
        Initialize the online ensemble model.
        
        Parameters:
        -----------
        models : list
            List of River models for online learning
        weights : array-like
            Initial weights for each model
        drift_detector : bool
            Whether to use concept drift detection
        feature_scaler : bool
            Whether to use feature scaling
        learning_rate : float
            Learning rate for weight updates
        """
        self.models = models or self._default_models()
        self.initial_weights = weights
        self.weights = weights if weights is not None else self._initialize_weights()
        self.drift_detector = drift_detector
        self.feature_scaler = feature_scaler
        self.learning_rate = learning_rate
        self.drift_detectors = None
        self.scalers = None
        self.models_fitted = False
        self.n_samples_seen = 0
        self.feature_names = None
        self.metrics = {
            'mae': metrics.MAE(),
            'rmse': metrics.RMSE(),
            'r2': metrics.R2()
        }
        
        # Initialize components
        self._initialize_components()
    
    def _default_models(self):
        """
        Create default online learning models.
        
        Returns:
        --------
        list
            List of default River models
        """
        models = [
            linear_model.LinearRegression(),
            linear_model.PARegressor(),
            linear_model.SGDRegressor(),
            linear_model.HoeffdingTreeRegressor()
        ]
        return models
    
    def _initialize_weights(self):
        """
        Initialize model weights.
        
        Returns:
        --------
        numpy.ndarray
            Initial weights for models
        """
        n_models = len(self.models)
        return np.ones(n_models) / n_models
    
    def _initialize_components(self):
        """
        Initialize drift detectors and scalers.
        """
        n_models = len(self.models)
        
        # Initialize drift detectors if enabled
        if self.drift_detector:
            self.drift_detectors = [drift.ADWIN() for _ in range(n_models)]
        
        # Initialize feature scalers if enabled
        if self.feature_scaler:
            self.scalers = [preprocessing.StandardScaler() for _ in range(n_models)]
    
    def partial_fit(self, X, y):
        """
        Update the model with new data.
        
        Parameters:
        -----------
        X : array-like
            New data
        y : array-like
            New target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Store feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame) and self.feature_names is None:
            self.feature_names = X.columns.tolist()
        
        # Convert to dictionaries for River models
        X_dict = self._convert_to_dict(X)
        
        # Update each model
        for i, model in enumerate(self.models):
            # Apply scaling if enabled
            if self.feature_scaler and self.scalers is not None:
                X_scaled = {k: self.scalers[i].learn_one({k: v}).transform_one({k: v})[k] 
                           for k, v in X_dict.items()}
            else:
                X_scaled = X_dict
            
            # Make prediction before updating
            y_pred = model.predict_one(X_scaled)
            
            # Update model
            model.learn_one(X_scaled, y)
            
            # Check for drift if enabled
            if self.drift_detector and self.drift_detectors is not None:
                if y_pred is not None:  # Only check if prediction was made
                    error = abs(y_pred - y)
                    drift_detected = self.drift_detectors[i].update(error)
                    
                    # Reset model if drift detected
                    if drift_detected:
                        # Create new model of same type
                        self.models[i] = type(model)()
                        
                        # Reset scaler if enabled
                        if self.feature_scaler and self.scalers is not None:
                            self.scalers[i] = preprocessing.StandardScaler()
            
            # Update metrics
            if y_pred is not None:
                for metric_name, metric in self.metrics.items():
                    metric.update(y, y_pred)
        
        # Update weights based on recent performance
        self._update_weights(X_dict, y)
        
        # Increment sample counter
        self.n_samples_seen += 1
        
        # Mark models as fitted
        self.models_fitted = True
        
        return self
    
    def fit(self, X, y):
        """
        Fit the model to data (batch version of partial_fit).
        
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
        # Reset model state
        self._initialize_components()
        self.weights = self.initial_weights if self.initial_weights is not None else self._initialize_weights()
        self.n_samples_seen = 0
        
        # Convert to DataFrames if needed
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        
        # Process data in small batches to simulate online learning
        batch_size = 1
        n_samples = len(X_df)
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X_df.iloc[i:end_idx]
            y_batch = y_series.iloc[i:end_idx]
            
            # Update with batch
            for j in range(len(X_batch)):
                self.partial_fit(X_batch.iloc[[j]], y_batch.iloc[j])
        
        return self
    
    def predict(self, X):
        """
        Generate predictions using the online ensemble model.
        
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
        
        # Convert to DataFrame if needed
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Initialize predictions array
        predictions = np.zeros(len(X_df))
        
        # Generate predictions for each sample
        for i in range(len(X_df)):
            X_dict = self._convert_to_dict(X_df.iloc[[i]])
            
            # Get predictions from each model
            model_predictions = []
            for j, model in enumerate(self.models):
                # Apply scaling if enabled
                if self.feature_scaler and self.scalers is not None:
                    X_scaled = {k: self.scalers[j].transform_one({k: v})[k] 
                               for k, v in X_dict.items()}
                else:
                    X_scaled = X_dict
                
                # Get prediction
                pred = model.predict_one(X_scaled)
                model_predictions.append(pred if pred is not None else 0)
            
            # Apply weights to predictions
            predictions[i] = np.sum(np.array(model_predictions) * self.weights)
        
        return predictions
    
    def predict_one(self, x):
        """
        Generate prediction for a single sample.
        
        Parameters:
        -----------
        x : dict or array-like
            Input sample
            
        Returns:
        --------
        float
            Predicted value
        """
        # Check if models are fitted
        if not self.models_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Convert to dict if needed
        x_dict = x if isinstance(x, dict) else self._convert_to_dict(x)
        
        # Get predictions from each model
        model_predictions = []
        for i, model in enumerate(self.models):
            # Apply scaling if enabled
            if self.feature_scaler and self.scalers is not None:
                x_scaled = {k: self.scalers[i].transform_one({k: v})[k] 
                           for k, v in x_dict.items()}
            else:
                x_scaled = x_dict
            
            # Get prediction
            pred = model.predict_one(x_scaled)
            model_predictions.append(pred if pred is not None else 0)
        
        # Apply weights to predictions
        return np.sum(np.array(model_predictions) * self.weights)
    
    def _convert_to_dict(self, X):
        """
        Convert input data to dictionary format for River models.
        
        Parameters:
        -----------
        X : array-like
            Input data
            
        Returns:
        --------
        dict
            Data in dictionary format
        """
        if isinstance(X, dict):
            return X
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                X_df = pd.DataFrame(X, columns=self.feature_names)
            else:
                X_df = pd.DataFrame(X)
        else:
            X_df = X
        
        # Handle single row
        if len(X_df) == 1:
            return {col: X_df[col].iloc[0] for col in X_df.columns}
        
        # Handle multiple rows (return first row)
        return {col: X_df[col].iloc[0] for col in X_df.columns}
    
    def _update_weights(self, X, y):
        """
        Update model weights based on performance.
        
        Parameters:
        -----------
        X : dict
            Input data
        y : float
            Target value
        """
        # Skip if too few samples
        if self.n_samples_seen < 10:
            return
        
        # Get predictions from each model
        errors = []
        for i, model in enumerate(self.models):
            # Apply scaling if enabled
            if self.feature_scaler and self.scalers is not None:
                X_scaled = {k: self.scalers[i].transform_one({k: v})[k] 
                           for k, v in X.items()}
            else:
                X_scaled = X
            
            # Get prediction
            pred = model.predict_one(X_scaled)
            
            # Calculate error
            if pred is not None:
                error = abs(pred - y)
                errors.append(error)
            else:
                errors.append(float('inf'))
        
        # Convert errors to weights (lower error = higher weight)
        if all(e == float('inf') for e in errors):
            # If all errors are infinite, use equal weights
            new_weights = np.ones(len(self.models)) / len(self.models)
        else:
            # Replace infinite errors with max finite error
            max_finite_error = max([e for e in errors if e != float('inf')], default=1.0)
            errors = [e if e != float('inf') else max_finite_error * 2 for e in errors]
            
            # Invert errors and normalize
            new_weights = 1.0 / (np.array(errors) + 1e-10)
            new_weights = new_weights / np.sum(new_weights)
        
        # Update weights with learning rate
        self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * new_weights
        
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
    
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
            'weights': self.initial_weights,
            'drift_detector': self.drift_detector,
            'feature_scaler': self.feature_scaler,
            'learning_rate': self.learning_rate
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
        
        # Re-initialize components if needed
        if 'models' in parameters or 'drift_detector' in parameters or 'feature_scaler' in parameters:
            self._initialize_components()
        
        # Update initial weights if weights parameter is set
        if 'weights' in parameters:
            self.initial_weights = parameters['weights']
            self.weights = self.initial_weights if self.initial_weights is not None else self._initialize_weights()
        
        return self
    
    def get_metrics(self):
        """
        Get current performance metrics.
        
        Returns:
        --------
        dict
            Dictionary of metric names and values
        """
        return {name: metric.get() for name, metric in self.metrics.items()}


class OnlineLearningManager:
    """
    Manager class for online learning models.
    Handles model training, updating, and persistence.
    """
    def __init__(self, model_dir='online_models'):
        """
        Initialize the online learning manager.
        
        Parameters:
        -----------
        model_dir : str
            Directory to store trained models
        """
        self.model_dir = model_dir
        self.models = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def create_model(self, model_name=None, **kwargs):
        """
        Create a new online learning model.
        
        Parameters:
        -----------
        model_name : str, optional
            Name for the model (generated if not provided)
        **kwargs : dict
            Additional parameters for the model
            
        Returns:
        --------
        str
            Name of the created model
        """
        # Generate model name if not provided
        if model_name is None:
            timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"online_{timestamp}"
        
        # Create model
        model = OnlineEnsembleModel(**kwargs)
        
        # Store model
        self.models[model_name] = model
        
        return model_name
    
    def update_model(self, model_name, X, y):
        """
        Update a model with new data.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to update
        X : array-like
            New data
        y : array-like
            New target values
            
        Returns:
        --------
        dict
            Updated metrics
        """
        # Get model
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model not found: {model_name}")
        
        # Update model
        model.partial_fit(X, y)
        
        # Return metrics
        return model.get_metrics()
    
    def predict(self, model_name, X):
        """
        Generate predictions using a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to use
        X : array-like
            Input data
            
        Returns:
        --------
        array-like
            Predicted values
        """
        # Get model
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model not found: {model_name}")
        
        # Generate predictions
        return model.predict(X)
    
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
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metrics
        metrics = model.get_metrics()
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
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Store model
        self.models[model_name] = model
        
        return model
    
    def list_models(self):
        """
        List all available models with their metrics.
        
        Returns:
        --------
        dict
            Dictionary of model names and their metrics
        """
        return {name: model.get_metrics() for name, model in self.models.items()}
    
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
        
        # Remove from disk
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)
        
        metrics_path = os.path.join(self.model_dir, f"{model_name}_metrics.json")
        if os.path.exists(metrics_path):
            os.remove(metrics_path)


class AdaptiveModelSelector:
    """
    Selects the best model for prediction based on recent performance.
    Combines batch-trained ensemble models with online learning models.
    """
    def __init__(self, ensemble_manager=None, online_manager=None, 
                 performance_window=100, adaptation_rate=0.1):
        """
        Initialize the adaptive model selector.
        
        Parameters:
        -----------
        ensemble_manager : ModelManager, optional
            Manager for batch-trained ensemble models
        online_manager : OnlineLearningManager, optional
            Manager for online learning models
        performance_window : int
            Number of recent predictions to consider for performance evaluation
        adaptation_rate : float
            Rate at which to adapt model selection weights
        """
        self.ensemble_manager = ensemble_manager
        self.online_manager = online_manager
        self.performance_window = performance_window
        self.adaptation_rate = adaptation_rate
        self.model_weights = {}
        self.recent_errors = {}
        self.n_predictions = 0
    
    def predict(self, X, default_model=None):
        """
        Generate predictions using the best model.
        
        Parameters:
        -----------
        X : array-like
            Input data
        default_model : str, optional
            Default model to use if no performance data is available
            
        Returns:
        --------
        array-like
            Predicted values
        dict
            Dictionary with model selection details
        """
        # Get available models
        ensemble_models = self.ensemble_manager.list_models() if self.ensemble_manager else {}
        online_models = self.online_manager.list_models() if self.online_manager else {}
        
        all_models = {**ensemble_models, **online_models}
        
        # If no models available, raise error
        if not all_models:
            raise ValueError("No models available for prediction")
        
        # If no performance data, use default model or first available
        if not self.model_weights:
            if default_model and default_model in all_models:
                selected_model = default_model
            else:
                selected_model = next(iter(all_models))
            
            # Initialize weights
            for model_name in all_models:
                self.model_weights[model_name] = 1.0 if model_name == selected_model else 0.0
                self.recent_errors[model_name] = []
        else:
            # Select model with highest weight
            selected_model = max(self.model_weights, key=self.model_weights.get)
        
        # Generate predictions using selected model
        if selected_model in ensemble_models:
            predictions = self.ensemble_manager.models[selected_model].predict(X)
        else:
            predictions = self.online_manager.models[selected_model].predict(X)
        
        # Return predictions and selection details
        details = {
            'selected_model': selected_model,
            'model_weights': self.model_weights,
            'n_predictions': self.n_predictions
        }
        
        return predictions, details
    
    def update_performance(self, model_name, error):
        """
        Update performance metrics for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        error : float
            Prediction error
        """
        # Initialize if model not in records
        if model_name not in self.recent_errors:
            self.recent_errors[model_name] = []
            self.model_weights[model_name] = 1.0 / len(self.model_weights) if self.model_weights else 1.0
        
        # Add error to recent errors
        self.recent_errors[model_name].append(error)
        
        # Keep only recent errors
        if len(self.recent_errors[model_name]) > self.performance_window:
            self.recent_errors[model_name].pop(0)
        
        # Update weights if enough data
        if all(len(errors) >= min(10, self.performance_window) for errors in self.recent_errors.values()):
            self._update_weights()
        
        # Increment prediction counter
        self.n_predictions += 1
    
    def _update_weights(self):
        """
        Update model weights based on recent performance.
        """
        # Calculate mean error for each model
        mean_errors = {model: np.mean(errors) for model, errors in self.recent_errors.items()}
        
        # Convert errors to weights (lower error = higher weight)
        if all(error == float('inf') for error in mean_errors.values()):
            # If all errors are infinite, use equal weights
            new_weights = {model: 1.0 / len(mean_errors) for model in mean_errors}
        else:
            # Replace infinite errors with max finite error
            max_finite_error = max([e for e in mean_errors.values() if e != float('inf')], default=1.0)
            mean_errors = {model: error if error != float('inf') else max_finite_error * 2 
                          for model, error in mean_errors.items()}
            
            # Invert errors and normalize
            weights_sum = sum(1.0 / (error + 1e-10) for error in mean_errors.values())
            new_weights = {model: (1.0 / (error + 1e-10)) / weights_sum 
                          for model, error in mean_errors.items()}
        
        # Update weights with adaptation rate
        for model in self.model_weights:
            self.model_weights[model] = ((1 - self.adaptation_rate) * self.model_weights[model] + 
                                        self.adaptation_rate * new_weights[model])
        
        # Normalize weights
        weights_sum = sum(self.model_weights.values())
        self.model_weights = {model: weight / weights_sum for model, weight in self.model_weights.items()}
    
    def get_best_model(self):
        """
        Get the currently best performing model.
        
        Returns:
        --------
        str
            Name of the best model
        float
            Weight of the best model
        """
        if not self.model_weights:
            return None, 0.0
        
        best_model = max(self.model_weights, key=self.model_weights.get)
        return best_model, self.model_weights[best_model]
