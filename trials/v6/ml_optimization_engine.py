import json
import logging
import math
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner

warnings.filterwarnings('ignore')


class PerformancePredictor:
    """Advanced performance prediction using multiple ML models"""
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_history = []
        self.is_trained = False
        
        # Initialize models
        self._initialize_models()
        
        logging.info(f"Performance Predictor initialized with {model_type} approach")
    
    def _initialize_models(self):
        """Initialize ML models for different metrics"""
        
        # Latency prediction models
        self.models['latency'] = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            'gaussian_process': GaussianProcessRegressor(
                kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
                random_state=42,
                n_restarts_optimizer=5
            ),
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        # Memory prediction models
        self.models['memory'] = {
            'random_forest': RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=4,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.15,
                max_depth=6,
                random_state=42
            ),
            'ridge': Ridge(alpha=0.5)
        }
        
        # Energy prediction models
        self.models['energy'] = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=80,
                learning_rate=0.2,
                max_depth=5,
                random_state=42
            )
        }
        
        # Accuracy prediction models
        self.models['accuracy'] = {
            'random_forest': RandomForestRegressor(
                n_estimators=120,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=60,
                learning_rate=0.25,
                max_depth=4,
                random_state=42
            )
        }
        
        # Initialize scalers
        for metric in self.models.keys():
            self.scalers[metric] = {
                'features': StandardScaler(),
                'target': RobustScaler()
            }
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        features = data.copy()
        
        # Encode categorical variables
        categorical_columns = ['model_name', 'precision', 'benchmark_type']
        
        for col in categorical_columns:
            if col in features.columns:
                # One-hot encoding
                dummies = pd.get_dummies(features[col], prefix=col)
                features = pd.concat([features, dummies], axis=1)
                features.drop(col, axis=1, inplace=True)
        
        # Create interaction features
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) >= 2:
            # Add polynomial features for key interactions
            for i, col1 in enumerate(numeric_columns[:3]):
                for col2 in numeric_columns[i+1:4]:
                    if col1 != col2:
                        features[f'{col1}_x_{col2}'] = features[col1] * features[col2]
        
        # Add derived features
        if 'num_parameters' in features.columns:
            features['log_num_parameters'] = np.log1p(features['num_parameters'])
        
        if 'model_size_mb' in features.columns:
            features['log_model_size'] = np.log1p(features['model_size_mb'])
        
        # Add time-based features if timestamp exists
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            features['hour'] = features['timestamp'].dt.hour
            features['day_of_week'] = features['timestamp'].dt.dayofweek
            features['month'] = features['timestamp'].dt.month
            features.drop('timestamp', axis=1, inplace=True)
        
        return features
    
    def train_models(self, data: pd.DataFrame, target_columns: List[str]):
        """Train prediction models for multiple metrics"""
        logging.info("Training performance prediction models...")
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Remove target columns from features
        feature_columns = [col for col in features.columns if col not in target_columns]
        X = features[feature_columns]
        
        training_results = {}
        
        for target_col in target_columns:
            if target_col not in data.columns:
                continue
            
            # Get target metric type
            metric_type = self._get_metric_type(target_col)
            
            if metric_type not in self.models:
                continue
            
            logging.info(f"Training models for {target_col} ({metric_type})")
            
            # Prepare target data
            y = data[target_col].dropna()
            
            # Align features and targets
            common_indices = X.index.intersection(y.index)
            X_aligned = X.loc[common_indices]
            y_aligned = y.loc[common_indices]
            
            if len(X_aligned) < 10:
                logging.warning(f"Insufficient data for {target_col}: {len(X_aligned)} samples")
                continue
            
            # Scale features and targets
            X_scaled = self.scalers[metric_type]['features'].fit_transform(X_aligned)
            y_scaled = self.scalers[metric_type]['target'].fit_transform(y_aligned.values.reshape(-1, 1)).ravel()
            
            # Train models
            model_results = {}
            
            for model_name, model in self.models[metric_type].items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_scaled, y_scaled, cv=5, scoring='r2')
                    
                    # Train on full dataset
                    model.fit(X_scaled, y_scaled)
                    
                    # Make predictions
                    y_pred_scaled = model.predict(X_scaled)
                    y_pred = self.scalers[metric_type]['target'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_aligned, y_pred)
                    mae = mean_absolute_error(y_aligned, y_pred)
                    r2 = r2_score(y_aligned, y_pred)
                    
                    model_results[model_name] = {
                        'cv_score_mean': cv_scores.mean(),
                        'cv_score_std': cv_scores.std(),
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'feature_importance': self._get_feature_importance(model, feature_columns)
                    }
                    
                    logging.info(f"  {model_name}: R² = {r2:.3f}, MAE = {mae:.3f}")
                    
                except Exception as e:
                    logging.error(f"Error training {model_name} for {target_col}: {e}")
            
            training_results[target_col] = model_results
            
            # Store feature importance for best model
            if model_results:
                best_model = max(model_results.keys(), key=lambda k: model_results[k]['r2'])
                self.feature_importance[target_col] = model_results[best_model]['feature_importance']
        
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'data_size': len(data),
            'features': len(feature_columns),
            'targets': len(target_columns),
            'results': training_results
        })
        
        self.is_trained = True
        logging.info("Model training completed")
        
        return training_results
    
    def predict_performance(self, features: Dict, target_metrics: List[str]) -> Dict:
        """Predict performance metrics"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = {}
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])
        feature_df = self.prepare_features(feature_df)
        
        for target_metric in target_metrics:
            metric_type = self._get_metric_type(target_metric)
            
            if metric_type not in self.models:
                continue
            
            try:
                # Scale features
                X_scaled = self.scalers[metric_type]['features'].transform(feature_df)
                
                # Get predictions from all models
                model_predictions = {}
                
                for model_name, model in self.models[metric_type].items():
                    y_pred_scaled = model.predict(X_scaled)
                    y_pred = self.scalers[metric_type]['target'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                    model_predictions[model_name] = y_pred[0]
                
                # Ensemble prediction (weighted average based on training performance)
                if len(model_predictions) > 1:
                    # Use equal weights for simplicity (could be improved with training scores)
                    ensemble_pred = np.mean(list(model_predictions.values()))
                    prediction_std = np.std(list(model_predictions.values()))
                else:
                    ensemble_pred = list(model_predictions.values())[0]
                    prediction_std = 0.0
                
                predictions[target_metric] = {
                    'prediction': ensemble_pred,
                    'uncertainty': prediction_std,
                    'individual_predictions': model_predictions
                }
                
            except Exception as e:
                logging.error(f"Error predicting {target_metric}: {e}")
                predictions[target_metric] = {
                    'prediction': None,
                    'uncertainty': None,
                    'error': str(e)
                }
        
        return predictions
    
    def _get_metric_type(self, target_column: str) -> str:
        """Determine metric type from column name"""
        if 'latency' in target_column.lower():
            return 'latency'
        elif 'memory' in target_column.lower():
            return 'memory'
        elif 'energy' in target_column.lower():
            return 'energy'
        elif 'accuracy' in target_column.lower():
            return 'accuracy'
        else:
            return 'latency'  # Default
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict:
        """Extract feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return {}
        
        return dict(zip(feature_names, importance))
    
    def save_models(self, filepath: str):
        """Save trained models to file"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']
        
        logging.info(f"Models loaded from {filepath}")


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna"""
    
    def __init__(self, optimization_direction: str = 'minimize'):
        self.optimization_direction = optimization_direction
        self.studies = {}
        self.optimization_history = []
        
        logging.info("Hyperparameter Optimizer initialized")
    
    def optimize_benchmark_config(self, objective_function: Callable, 
                                config_space: Dict, n_trials: int = 100) -> Dict:
        """Optimize benchmark configuration parameters"""
        
        study_name = f"benchmark_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create study
        study = optuna.create_study(
            direction=self.optimization_direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        def optuna_objective(trial):
            # Sample parameters from config space
            params = {}
            
            for param_name, param_config in config_space.items():
                param_type = param_config['type']
                
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config['low'], 
                        param_config['high'],
                        step=param_config.get('step', 1)
                    )
                elif param_type == 'float':
                    if param_config.get('log', False):
                        params[param_name] = trial.suggest_loguniform(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    else:
                        params[param_name] = trial.suggest_uniform(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            
            # Evaluate objective function
            try:
                result = objective_function(params)
                
                # Handle multi-objective optimization
                if isinstance(result, dict):
                    # Use weighted sum for multi-objective
                    weights = config_space.get('objective_weights', {})
                    weighted_result = 0
                    
                    for metric, value in result.items():
                        weight = weights.get(metric, 1.0)
                        weighted_result += weight * value
                    
                    return weighted_result
                else:
                    return result
                    
            except Exception as e:
                logging.error(f"Objective function error: {e}")
                # Return worst possible value
                return float('inf') if self.optimization_direction == 'minimize' else float('-inf')
        
        # Run optimization
        logging.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        study.optimize(optuna_objective, n_trials=n_trials, timeout=3600)  # 1 hour timeout
        
        # Store results
        self.studies[study_name] = study
        
        optimization_result = {
            'study_name': study_name,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'optimization_history': [
                {
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in study.trials
            ]
        }
        
        self.optimization_history.append(optimization_result)
        
        logging.info(f"Optimization completed. Best value: {study.best_value:.4f}")
        logging.info(f"Best parameters: {study.best_params}")
        
        return optimization_result
    
    def multi_objective_optimization(self, objective_functions: Dict[str, Callable],
                                   config_space: Dict, n_trials: int = 100) -> Dict:
        """Multi-objective optimization using Pareto efficiency"""
        
        study_name = f"multi_objective_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create multi-objective study
        study = optuna.create_study(
            directions=['minimize'] * len(objective_functions),
            sampler=TPESampler(seed=42)
        )
        
        def multi_objective(trial):
            # Sample parameters
            params = {}
            
            for param_name, param_config in config_space.items():
                param_type = param_config['type']
                
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config['low'], 
                        param_config['high']
                    )
                elif param_type == 'float':
                    params[param_name] = trial.suggest_uniform(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            
            # Evaluate all objectives
            results = []
            
            for obj_name, obj_func in objective_functions.items():
                try:
                    result = obj_func(params)
                    results.append(result)
                except Exception as e:
                    logging.error(f"Objective {obj_name} error: {e}")
                    results.append(float('inf'))
            
            return results
        
        # Run optimization
        logging.info(f"Starting multi-objective optimization with {n_trials} trials")
        
        study.optimize(multi_objective, n_trials=n_trials)
        
        # Find Pareto front
        pareto_trials = []
        
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                is_pareto = True
                
                for other_trial in study.trials:
                    if (other_trial.state == optuna.trial.TrialState.COMPLETE and 
                        other_trial != trial):
                        
                        # Check if other_trial dominates trial
                        dominates = True
                        for i in range(len(trial.values)):
                            if other_trial.values[i] >= trial.values[i]:
                                dominates = False
                                break
                        
                        if dominates:
                            is_pareto = False
                            break
                
                if is_pareto:
                    pareto_trials.append(trial)
        
        # Store results
        self.studies[study_name] = study
        
        result = {
            'study_name': study_name,
            'pareto_front': [
                {
                    'params': trial.params,
                    'values': trial.values,
                    'trial_number': trial.number
                }
                for trial in pareto_trials
            ],
            'n_trials': len(study.trials),
            'n_pareto_solutions': len(pareto_trials)
        }
        
        logging.info(f"Multi-objective optimization completed")
        logging.info(f"Found {len(pareto_trials)} Pareto optimal solutions")
        
        return result


class IntelligentScheduler:
    """Intelligent task scheduling using ML predictions"""
    
    def __init__(self, performance_predictor: PerformancePredictor):
        self.performance_predictor = performance_predictor
        self.scheduling_history = []
        self.resource_constraints = {}
        
        logging.info("Intelligent Scheduler initialized")
    
    def set_resource_constraints(self, constraints: Dict):
        """Set resource constraints for scheduling"""
        self.resource_constraints = constraints
        logging.info(f"Resource constraints updated: {constraints}")
    
    def schedule_tasks(self, tasks: List[Dict], workers: List[Dict]) -> List[Dict]:
        """Schedule tasks to workers using ML predictions"""
        
        scheduled_tasks = []
        
        # Predict performance for each task-worker combination
        task_predictions = {}
        
        for task in tasks:
            task_id = task['id']
            task_predictions[task_id] = {}
            
            for worker in workers:
                worker_id = worker['id']
                
                # Create feature vector for prediction
                features = {
                    'model_name': task.get('model_name', 'unknown'),
                    'precision': task.get('precision', 'fp16'),
                    'benchmark_type': task.get('benchmark_type', 'latency'),
                    'worker_type': worker.get('type', 'gpu'),
                    'worker_memory_gb': worker.get('memory_gb', 16),
                    'worker_gpu_count': worker.get('gpu_count', 1),
                    'current_load': worker.get('current_load', 0.0)
                }
                
                # Predict performance metrics
                predictions = self.performance_predictor.predict_performance(
                    features, ['latency_mean', 'memory_peak_mb', 'energy_joules']
                )
                
                task_predictions[task_id][worker_id] = predictions
        
        # Optimize task assignment
        assignment = self._optimize_assignment(tasks, workers, task_predictions)
        
        # Create scheduled task list
        for task_id, worker_id in assignment.items():
            task = next(t for t in tasks if t['id'] == task_id)
            worker = next(w for w in workers if w['id'] == worker_id)
            
            scheduled_task = {
                'task_id': task_id,
                'worker_id': worker_id,
                'predicted_latency': task_predictions[task_id][worker_id].get('latency_mean', {}).get('prediction', 0),
                'predicted_memory': task_predictions[task_id][worker_id].get('memory_peak_mb', {}).get('prediction', 0),
                'predicted_energy': task_predictions[task_id][worker_id].get('energy_joules', {}).get('prediction', 0),
                'priority': task.get('priority', 1),
                'scheduled_at': datetime.now().isoformat()
            }
            
            scheduled_tasks.append(scheduled_task)
        
        # Record scheduling decision
        self.scheduling_history.append({
            'timestamp': datetime.now().isoformat(),
            'n_tasks': len(tasks),
            'n_workers': len(workers),
            'assignments': assignment,
            'total_predicted_time': sum(
                task_predictions[task_id][worker_id].get('latency_mean', {}).get('prediction', 0)
                for task_id, worker_id in assignment.items()
            )
        })
        
        logging.info(f"Scheduled {len(scheduled_tasks)} tasks across {len(workers)} workers")
        
        return scheduled_tasks
    
    def _optimize_assignment(self, tasks: List[Dict], workers: List[Dict], 
                           predictions: Dict) -> Dict[str, str]:
        """Optimize task-to-worker assignment"""
        
        # Simple greedy assignment for now (could be improved with more sophisticated algorithms)
        assignment = {}
        worker_loads = {worker['id']: 0.0 for worker in workers}
        
        # Sort tasks by priority (higher priority first)
        sorted_tasks = sorted(tasks, key=lambda t: t.get('priority', 1), reverse=True)
        
        for task in sorted_tasks:
            task_id = task['id']
            best_worker = None
            best_score = float('inf')
            
            for worker in workers:
                worker_id = worker['id']
                
                # Check resource constraints
                if not self._check_resource_constraints(task, worker, worker_loads[worker_id]):
                    continue
                
                # Calculate assignment score (lower is better)
                pred = predictions[task_id][worker_id]
                
                latency = pred.get('latency_mean', {}).get('prediction', 1.0)
                memory = pred.get('memory_peak_mb', {}).get('prediction', 1000.0)
                energy = pred.get('energy_joules', {}).get('prediction', 100.0)
                
                # Weighted score considering multiple objectives
                score = (
                    0.4 * latency +  # 40% weight on latency
                    0.3 * (memory / 1000.0) +  # 30% weight on memory (normalized)
                    0.2 * (energy / 100.0) +  # 20% weight on energy (normalized)
                    0.1 * worker_loads[worker_id]  # 10% weight on current load
                )
                
                if score < best_score:
                    best_score = score
                    best_worker = worker_id
            
            if best_worker:
                assignment[task_id] = best_worker
                # Update worker load
                pred_latency = predictions[task_id][best_worker].get('latency_mean', {}).get('prediction', 1.0)
                worker_loads[best_worker] += pred_latency
            else:
                # No suitable worker found, assign to least loaded worker
                least_loaded_worker = min(worker_loads.keys(), key=lambda w: worker_loads[w])
                assignment[task_id] = least_loaded_worker
                worker_loads[least_loaded_worker] += 1.0  # Default load increment
        
        return assignment
    
    def _check_resource_constraints(self, task: Dict, worker: Dict, current_load: float) -> bool:
        """Check if task can be assigned to worker given resource constraints"""
        
        # Check memory constraints
        max_memory = self.resource_constraints.get('max_memory_per_worker', float('inf'))
        worker_memory = worker.get('memory_gb', 16) * 1024  # Convert to MB
        
        if worker_memory > max_memory:
            return False
        
        # Check load constraints
        max_load = self.resource_constraints.get('max_load_per_worker', float('inf'))
        
        if current_load > max_load:
            return False
        
        # Check worker type compatibility
        required_worker_type = task.get('required_worker_type')
        worker_type = worker.get('type')
        
        if required_worker_type and worker_type != required_worker_type:
            return False
        
        return True
    
    def get_scheduling_statistics(self) -> Dict:
        """Get scheduling performance statistics"""
        if not self.scheduling_history:
            return {}
        
        recent_schedules = self.scheduling_history[-10:]  # Last 10 scheduling decisions
        
        avg_tasks_per_schedule = np.mean([s['n_tasks'] for s in recent_schedules])
        avg_workers_per_schedule = np.mean([s['n_workers'] for s in recent_schedules])
        avg_predicted_time = np.mean([s['total_predicted_time'] for s in recent_schedules])
        
        return {
            'total_scheduling_decisions': len(self.scheduling_history),
            'avg_tasks_per_schedule': avg_tasks_per_schedule,
            'avg_workers_per_schedule': avg_workers_per_schedule,
            'avg_predicted_total_time': avg_predicted_time,
            'resource_constraints': self.resource_constraints
        }


class AutoMLOptimizer:
    """Automated ML pipeline for benchmark optimization"""
    
    def __init__(self, output_dir: str = "ml_optimization_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.performance_predictor = PerformancePredictor()
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.intelligent_scheduler = IntelligentScheduler(self.performance_predictor)
        
        self.optimization_history = []
        
        logging.info(f"AutoML Optimizer initialized")
        logging.info(f"Output directory: {self.output_dir}")
    
    def train_performance_models(self, benchmark_data: pd.DataFrame) -> Dict:
        """Train performance prediction models"""
        logging.info("Training performance prediction models...")
        
        # Define target metrics
        target_metrics = [
            col for col in benchmark_data.columns
            if any(keyword in col.lower() for keyword in ['latency', 'memory', 'energy', 'accuracy'])
        ]
        
        if not target_metrics:
            raise ValueError("No performance metrics found in data")
        
        # Train models
        training_results = self.performance_predictor.train_models(benchmark_data, target_metrics)
        
        # Save models
        model_file = self.output_dir / "performance_models.pkl"
        self.performance_predictor.save_models(str(model_file))
        
        # Save training results
        results_file = self.output_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        logging.info(f"Performance models trained and saved to {model_file}")
        
        return training_results
    
    def optimize_benchmark_configuration(self, objective_metrics: List[str], 
                                       config_space: Dict, n_trials: int = 100) -> Dict:
        """Optimize benchmark configuration using hyperparameter optimization"""
        
        def objective_function(params):
            # Simulate benchmark execution with given parameters
            # In practice, this would run actual benchmarks
            
            # Create feature vector from parameters
            features = {
                'model_name': params.get('model_name', 'default_model'),
                'precision': params.get('precision', 'fp16'),
                'benchmark_type': params.get('benchmark_type', 'latency'),
                'num_runs': params.get('num_runs', 100),
                'batch_size': params.get('batch_size', 8),
                'max_tokens': params.get('max_tokens', 32)
            }
            
            # Predict performance
            predictions = self.performance_predictor.predict_performance(features, objective_metrics)
            
            # Calculate objective value (minimize latency, memory, energy; maximize accuracy)
            objective_value = 0
            
            for metric in objective_metrics:
                pred = predictions.get(metric, {}).get('prediction', 0)
                
                if 'accuracy' in metric.lower():
                    objective_value -= pred  # Maximize accuracy (minimize negative accuracy)
                else:
                    objective_value += pred  # Minimize latency, memory, energy
            
            return objective_value
        
        # Run optimization
        optimization_result = self.hyperparameter_optimizer.optimize_benchmark_config(
            objective_function, config_space, n_trials
        )
        
        # Save results
        results_file = self.output_dir / f"optimization_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(optimization_result, f, indent=2, default=str)
        
        self.optimization_history.append(optimization_result)
        
        logging.info(f"Configuration optimization completed. Results saved to {results_file}")
        
        return optimization_result
    
    def multi_objective_benchmark_optimization(self, objectives: Dict[str, str],
                                             config_space: Dict, n_trials: int = 100) -> Dict:
        """Multi-objective optimization for benchmark configuration"""
        
        objective_functions = {}
        
        for obj_name, metric_name in objectives.items():
            def create_objective(metric):
                def objective_func(params):
                    features = {
                        'model_name': params.get('model_name', 'default_model'),
                        'precision': params.get('precision', 'fp16'),
                        'benchmark_type': params.get('benchmark_type', 'latency'),
                        **params
                    }
                    
                    predictions = self.performance_predictor.predict_performance(features, [metric])
                    return predictions.get(metric, {}).get('prediction', 0)
                
                return objective_func
            
            objective_functions[obj_name] = create_objective(metric_name)
        
        # Run multi-objective optimization
        result = self.hyperparameter_optimizer.multi_objective_optimization(
            objective_functions, config_space, n_trials
        )
        
        # Save results
        results_file = self.output_dir / f"multi_objective_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logging.info(f"Multi-objective optimization completed. Results saved to {results_file}")
        
        return result
    
    def intelligent_task_scheduling(self, tasks: List[Dict], workers: List[Dict],
                                  resource_constraints: Dict = None) -> List[Dict]:
        """Perform intelligent task scheduling"""
        
        if resource_constraints:
            self.intelligent_scheduler.set_resource_constraints(resource_constraints)
        
        scheduled_tasks = self.intelligent_scheduler.schedule_tasks(tasks, workers)
        
        # Save scheduling results
        scheduling_file = self.output_dir / f"scheduling_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(scheduling_file, 'w') as f:
            json.dump(scheduled_tasks, f, indent=2, default=str)
        
        logging.info(f"Task scheduling completed. {len(scheduled_tasks)} tasks scheduled")
        
        return scheduled_tasks
    
    def generate_optimization_report(self) -> Dict:
        """Generate comprehensive optimization report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_predictor': {
                'is_trained': self.performance_predictor.is_trained,
                'training_history': self.performance_predictor.training_history,
                'feature_importance': self.performance_predictor.feature_importance
            },
            'hyperparameter_optimization': {
                'optimization_history': self.hyperparameter_optimizer.optimization_history,
                'n_studies': len(self.hyperparameter_optimizer.studies)
            },
            'intelligent_scheduling': {
                'statistics': self.intelligent_scheduler.get_scheduling_statistics()
            },
            'optimization_summary': self._generate_optimization_summary()
        }
        
        # Save report
        report_file = self.output_dir / "optimization_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logging.info(f"Optimization report generated: {report_file}")
        
        return report
    
    def _generate_optimization_summary(self) -> Dict:
        """Generate optimization summary"""
        
        summary = {
            'total_optimizations': len(self.optimization_history),
            'best_configurations': [],
            'optimization_trends': {}
        }
        
        if self.optimization_history:
            # Find best configurations
            for opt_result in self.optimization_history[-5:]:  # Last 5 optimizations
                summary['best_configurations'].append({
                    'study_name': opt_result['study_name'],
                    'best_value': opt_result['best_value'],
                    'best_params': opt_result['best_params']
                })
            
            # Analyze optimization trends
            best_values = [opt['best_value'] for opt in self.optimization_history]
            
            if len(best_values) > 1:
                # Calculate improvement trend
                improvements = [best_values[i] - best_values[i-1] for i in range(1, len(best_values))]
                
                summary['optimization_trends'] = {
                    'avg_improvement': np.mean(improvements),
                    'total_improvement': best_values[-1] - best_values[0],
                    'improvement_rate': np.mean(improvements) / abs(best_values[0]) if best_values[0] != 0 else 0
                }
        
        return summary


def main():
    """Main ML optimization demo"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create optimizer
    optimizer = AutoMLOptimizer()
    
    # Generate sample benchmark data
    sample_data = generate_sample_benchmark_data()
    
    print("Sample data shape:", sample_data.shape)
    print("Columns:", sample_data.columns.tolist())
    
    # Train performance models
    training_results = optimizer.train_performance_models(sample_data)
    
    print("\nTraining Results:")
    for target, results in training_results.items():
        print(f"  {target}:")
        for model, metrics in results.items():
            print(f"    {model}: R² = {metrics['r2']:.3f}")
    
    # Define optimization configuration space
    config_space = {
        'model_name': {
            'type': 'categorical',
            'choices': ['model_small', 'model_medium', 'model_large']
        },
        'precision': {
            'type': 'categorical',
            'choices': ['fp16', 'int8', 'int4']
        },
        'num_runs': {
            'type': 'int',
            'low': 50,
            'high': 200
        },
        'batch_size': {
            'type': 'int',
            'low': 1,
            'high': 16
        },
        'max_tokens': {
            'type': 'int',
            'low': 16,
            'high': 64
        }
    }
    
    # Optimize benchmark configuration
    optimization_result = optimizer.optimize_benchmark_configuration(
        objective_metrics=['latency_mean', 'memory_peak_mb'],
        config_space=config_space,
        n_trials=50
    )
    
    print(f"\nOptimization Results:")
    print(f"  Best value: {optimization_result['best_value']:.4f}")
    print(f"  Best parameters: {optimization_result['best_params']}")
    
    # Multi-objective optimization
    multi_obj_result = optimizer.multi_objective_benchmark_optimization(
        objectives={
            'latency': 'latency_mean',
            'memory': 'memory_peak_mb',
            'energy': 'energy_joules'
        },
        config_space=config_space,
        n_trials=30
    )
    
    print(f"\nMulti-objective Results:")
    print(f"  Pareto solutions found: {multi_obj_result['n_pareto_solutions']}")
    
    # Intelligent scheduling demo
    sample_tasks = [
        {'id': f'task_{i}', 'model_name': 'model_medium', 'precision': 'fp16', 'benchmark_type': 'latency', 'priority': i % 3}
        for i in range(10)
    ]
    
    sample_workers = [
        {'id': f'worker_{i}', 'type': 'gpu', 'memory_gb': 16, 'gpu_count': 1, 'current_load': 0.0}
        for i in range(3)
    ]
    
    scheduled_tasks = optimizer.intelligent_task_scheduling(
        sample_tasks, 
        sample_workers,
        resource_constraints={'max_memory_per_worker': 20000, 'max_load_per_worker': 5.0}
    )
    
    print(f"\nScheduling Results:")
    print(f"  Tasks scheduled: {len(scheduled_tasks)}")
    
    for task in scheduled_tasks[:5]:  # Show first 5
        print(f"    Task {task['task_id']} -> Worker {task['worker_id']} "
              f"(predicted latency: {task['predicted_latency']:.3f}s)")
    
    # Generate comprehensive report
    report = optimizer.generate_optimization_report()
    
    print(f"\nOptimization Report Generated:")
    print(f"  Total optimizations: {report['optimization_summary']['total_optimizations']}")
    print(f"  Performance models trained: {report['performance_predictor']['is_trained']}")
    
    print(f"\nResults saved to: {optimizer.output_dir}")


def generate_sample_benchmark_data() -> pd.DataFrame:
    """Generate sample benchmark data for ML training"""
    np.random.seed(42)
    
    models = ['model_small', 'model_medium', 'model_large']
    precisions = ['fp16', 'int8', 'int4']
    benchmark_types = ['latency', 'memory', 'energy', 'accuracy']
    
    data = []
    
    for i in range(500):  # Generate 500 samples
        model = np.random.choice(models)
        precision = np.random.choice(precisions)
        benchmark_type = np.random.choice(benchmark_types)
        
        # Simulate realistic relationships
        model_factor = {'model_small': 0.5, 'model_medium': 1.0, 'model_large': 2.0}[model]
        precision_factor = {'fp16': 1.0, 'int8': 0.7, 'int4': 0.5}[precision]
        
        # Add some noise and correlations
        base_latency = 0.1 * model_factor / precision_factor
        base_memory = 1000 * model_factor * precision_factor
        base_energy = 50 * model_factor / precision_factor
        base_accuracy = 0.8 - (1 - precision_factor) * 0.1
        
        record = {
            'model_name': model,
            'precision': precision,
            'benchmark_type': benchmark_type,
            'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 30)),
            'num_parameters': int(model_factor * 1e9),  # Parameters in billions
            'model_size_mb': base_memory * 0.8,
            'latency_mean': max(0.01, np.random.normal(base_latency, base_latency * 0.2)),
            'latency_p95': max(0.01, np.random.normal(base_latency * 1.5, base_latency * 0.3)),
            'memory_peak_mb': max(100, np.random.normal(base_memory, base_memory * 0.15)),
            'energy_joules': max(1, np.random.normal(base_energy, base_energy * 0.25)),
            'accuracy_score': max(0.0, min(1.0, np.random.normal(base_accuracy, 0.05))),
            'throughput_tokens_per_sec': max(1, np.random.normal(100 / model_factor * precision_factor, 20))
        }
        
        data.append(record)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()