import json
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """Advanced statistical analysis for benchmark data"""
    
    def __init__(self):
        self.confidence_level = 0.95
        self.significance_level = 0.05
    
    def descriptive_statistics(self, data: List[float]) -> Dict:
        """Calculate comprehensive descriptive statistics"""
        if not data:
            return {}
        
        data_array = np.array(data)
        
        # Basic statistics
        stats_dict = {
            'count': len(data),
            'mean': np.mean(data_array),
            'median': np.median(data_array),
            'mode': stats.mode(data_array, keepdims=True)[0][0] if len(data) > 1 else data[0],
            'std': np.std(data_array, ddof=1),
            'variance': np.var(data_array, ddof=1),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'range': np.max(data_array) - np.min(data_array),
            'q1': np.percentile(data_array, 25),
            'q3': np.percentile(data_array, 75),
            'iqr': np.percentile(data_array, 75) - np.percentile(data_array, 25),
            'skewness': stats.skew(data_array),
            'kurtosis': stats.kurtosis(data_array),
            'coefficient_of_variation': np.std(data_array) / np.mean(data_array) if np.mean(data_array) != 0 else 0
        }
        
        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            stats_dict[f'p{p}'] = np.percentile(data_array, p)
        
        # Confidence intervals
        confidence_interval = stats.t.interval(
            self.confidence_level,
            len(data) - 1,
            loc=np.mean(data_array),
            scale=stats.sem(data_array)
        )
        stats_dict['confidence_interval_lower'] = confidence_interval[0]
        stats_dict['confidence_interval_upper'] = confidence_interval[1]
        
        # Outlier detection using IQR method
        q1, q3 = stats_dict['q1'], stats_dict['q3']
        iqr = stats_dict['iqr']
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data_array[(data_array < lower_bound) | (data_array > upper_bound)]
        stats_dict['outlier_count'] = len(outliers)
        stats_dict['outlier_percentage'] = len(outliers) / len(data) * 100
        
        return stats_dict
    
    def normality_test(self, data: List[float]) -> Dict:
        """Test for normality using multiple methods"""
        if len(data) < 3:
            return {'error': 'Insufficient data for normality testing'}
        
        data_array = np.array(data)
        
        results = {}
        
        # Shapiro-Wilk test (best for small samples)
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data_array)
            results['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > self.significance_level
            }
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data_array, 'norm', args=(np.mean(data_array), np.std(data_array)))
        results['kolmogorov_smirnov'] = {
            'statistic': ks_stat,
            'p_value': ks_p,
            'is_normal': ks_p > self.significance_level
        }
        
        # Anderson-Darling test
        ad_stat, ad_critical, ad_significance = stats.anderson(data_array, dist='norm')
        results['anderson_darling'] = {
            'statistic': ad_stat,
            'critical_values': ad_critical.tolist(),
            'significance_levels': ad_significance.tolist(),
            'is_normal': ad_stat < ad_critical[2]  # 5% significance level
        }
        
        # D'Agostino's normality test
        if len(data) >= 8:
            dagostino_stat, dagostino_p = stats.normaltest(data_array)
            results['dagostino'] = {
                'statistic': dagostino_stat,
                'p_value': dagostino_p,
                'is_normal': dagostino_p > self.significance_level
            }
        
        return results
    
    def correlation_analysis(self, x_data: List[float], y_data: List[float]) -> Dict:
        """Comprehensive correlation analysis"""
        if len(x_data) != len(y_data) or len(x_data) < 3:
            return {'error': 'Invalid data for correlation analysis'}
        
        x_array = np.array(x_data)
        y_array = np.array(y_data)
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(x_array, y_array)
        
        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(x_array, y_array)
        
        # Kendall's tau
        kendall_tau, kendall_p = stats.kendalltau(x_array, y_array)
        
        return {
            'pearson': {
                'correlation': pearson_r,
                'p_value': pearson_p,
                'significant': pearson_p < self.significance_level,
                'strength': self._interpret_correlation_strength(abs(pearson_r))
            },
            'spearman': {
                'correlation': spearman_r,
                'p_value': spearman_p,
                'significant': spearman_p < self.significance_level,
                'strength': self._interpret_correlation_strength(abs(spearman_r))
            },
            'kendall': {
                'correlation': kendall_tau,
                'p_value': kendall_p,
                'significant': kendall_p < self.significance_level,
                'strength': self._interpret_correlation_strength(abs(kendall_tau))
            }
        }
    
    def _interpret_correlation_strength(self, r: float) -> str:
        """Interpret correlation strength"""
        if r < 0.1:
            return 'negligible'
        elif r < 0.3:
            return 'weak'
        elif r < 0.5:
            return 'moderate'
        elif r < 0.7:
            return 'strong'
        else:
            return 'very_strong'
    
    def hypothesis_testing(self, group1: List[float], group2: List[float], 
                          test_type: str = 'auto') -> Dict:
        """Perform hypothesis testing between two groups"""
        if len(group1) < 2 or len(group2) < 2:
            return {'error': 'Insufficient data for hypothesis testing'}
        
        group1_array = np.array(group1)
        group2_array = np.array(group2)
        
        results = {}
        
        # Determine appropriate test
        if test_type == 'auto':
            # Check normality and equal variances
            _, p1 = stats.shapiro(group1_array) if len(group1) <= 5000 else (0, 0.1)
            _, p2 = stats.shapiro(group2_array) if len(group2) <= 5000 else (0, 0.1)
            
            normal1 = p1 > self.significance_level
            normal2 = p2 > self.significance_level
            
            if normal1 and normal2:
                # Check equal variances
                _, levene_p = stats.levene(group1_array, group2_array)
                equal_var = levene_p > self.significance_level
                test_type = 'ttest' if equal_var else 'welch_ttest'
            else:
                test_type = 'mannwhitney'
        
        # Perform appropriate test
        if test_type == 'ttest':
            stat, p_value = stats.ttest_ind(group1_array, group2_array, equal_var=True)
            test_name = "Independent t-test"
        elif test_type == 'welch_ttest':
            stat, p_value = stats.ttest_ind(group1_array, group2_array, equal_var=False)
            test_name = "Welch's t-test"
        elif test_type == 'mannwhitney':
            stat, p_value = stats.mannwhitneyu(group1_array, group2_array, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        else:
            return {'error': f'Unknown test type: {test_type}'}
        
        # Effect size calculation
        effect_size = self._calculate_effect_size(group1_array, group2_array, test_type)
        
        results = {
            'test_name': test_name,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'effect_size': effect_size,
            'group1_mean': np.mean(group1_array),
            'group2_mean': np.mean(group2_array),
            'difference': np.mean(group1_array) - np.mean(group2_array)
        }
        
        return results
    
    def _calculate_effect_size(self, group1: np.ndarray, group2: np.ndarray, test_type: str) -> Dict:
        """Calculate effect size measures"""
        if test_type in ['ttest', 'welch_ttest']:
            # Cohen's d
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                 (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                (len(group1) + len(group2) - 2))
            cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
            
            return {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_cohens_d(abs(cohens_d))
            }
        elif test_type == 'mannwhitney':
            # Rank-biserial correlation
            n1, n2 = len(group1), len(group2)
            u_stat, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            r = 1 - (2 * u_stat) / (n1 * n2)
            
            return {
                'rank_biserial_correlation': r,
                'interpretation': self._interpret_correlation_strength(abs(r))
            }
        
        return {}
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return 'negligible'
        elif d < 0.5:
            return 'small'
        elif d < 0.8:
            return 'medium'
        else:
            return 'large'


class TrendAnalyzer:
    """Time series and trend analysis"""
    
    def __init__(self):
        self.min_points = 3
    
    def detect_trends(self, timestamps: List[str], values: List[float]) -> Dict:
        """Detect trends in time series data"""
        if len(timestamps) != len(values) or len(values) < self.min_points:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Convert timestamps to numeric values
        try:
            time_values = [datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp() 
                          for ts in timestamps]
        except:
            # Fallback: use indices if timestamp parsing fails
            time_values = list(range(len(timestamps)))
        
        time_array = np.array(time_values)
        value_array = np.array(values)
        
        # Linear trend analysis
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_array, value_array)
        
        # Mann-Kendall trend test
        mk_result = self._mann_kendall_test(value_array)
        
        # Seasonal decomposition (if enough data points)
        seasonal_analysis = {}
        if len(values) >= 12:
            seasonal_analysis = self._seasonal_decomposition(value_array)
        
        # Change point detection
        change_points = self._detect_change_points(value_array)
        
        return {
            'linear_trend': {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            },
            'mann_kendall': mk_result,
            'seasonal_analysis': seasonal_analysis,
            'change_points': change_points,
            'trend_strength': self._calculate_trend_strength(value_array)
        }
    
    def _mann_kendall_test(self, data: np.ndarray) -> Dict:
        """Mann-Kendall trend test"""
        n = len(data)
        
        # Calculate S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data[j] > data[i]:
                    s += 1
                elif data[j] < data[i]:
                    s -= 1
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            'statistic': s,
            'z_score': z,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'trend': 'increasing' if z > 0 else 'decreasing' if z < 0 else 'no_trend'
        }
    
    def _seasonal_decomposition(self, data: np.ndarray) -> Dict:
        """Simple seasonal decomposition"""
        # This is a simplified version - in practice, you'd use statsmodels
        n = len(data)
        period = min(12, n // 3)  # Assume monthly seasonality or adjust based on data
        
        if period < 2:
            return {'error': 'Insufficient data for seasonal analysis'}
        
        # Calculate seasonal indices
        seasonal_indices = []
        for i in range(period):
            season_values = data[i::period]
            if len(season_values) > 0:
                seasonal_indices.append(np.mean(season_values))
            else:
                seasonal_indices.append(0)
        
        # Deseasonalize data
        deseasonalized = []
        for i, value in enumerate(data):
            seasonal_index = seasonal_indices[i % period]
            if seasonal_index != 0:
                deseasonalized.append(value / seasonal_index)
            else:
                deseasonalized.append(value)
        
        return {
            'period': period,
            'seasonal_indices': seasonal_indices,
            'seasonality_strength': np.std(seasonal_indices) / np.mean(seasonal_indices) if np.mean(seasonal_indices) != 0 else 0
        }
    
    def _detect_change_points(self, data: np.ndarray) -> List[int]:
        """Simple change point detection using cumulative sum"""
        if len(data) < 6:
            return []
        
        # Calculate cumulative sum of deviations from mean
        mean_val = np.mean(data)
        cusum = np.cumsum(data - mean_val)
        
        # Find points where cumulative sum changes direction significantly
        change_points = []
        threshold = 2 * np.std(data)
        
        for i in range(2, len(cusum) - 2):
            if (abs(cusum[i] - cusum[i-1]) > threshold and
                abs(cusum[i+1] - cusum[i]) > threshold and
                np.sign(cusum[i] - cusum[i-1]) != np.sign(cusum[i+1] - cusum[i])):
                change_points.append(i)
        
        return change_points
    
    def _calculate_trend_strength(self, data: np.ndarray) -> float:
        """Calculate overall trend strength"""
        if len(data) < 3:
            return 0.0
        
        # Use coefficient of variation of first differences
        first_diff = np.diff(data)
        if np.mean(first_diff) == 0:
            return 0.0
        
        cv = np.std(first_diff) / abs(np.mean(first_diff))
        # Convert to 0-1 scale (lower CV = stronger trend)
        return max(0, 1 - cv / 2)


class ClusterAnalyzer:
    """Clustering analysis for benchmark data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def perform_clustering(self, data: pd.DataFrame, features: List[str], 
                          n_clusters: Optional[int] = None) -> Dict:
        """Perform clustering analysis on benchmark data"""
        if data.empty or not features:
            return {'error': 'No data or features provided'}
        
        # Prepare data
        feature_data = data[features].copy()
        
        # Handle categorical variables
        processed_features = []
        for feature in features:
            if feature_data[feature].dtype == 'object':
                # Encode categorical variables
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                feature_data[feature] = self.label_encoders[feature].fit_transform(feature_data[feature].astype(str))
            processed_features.append(feature)
        
        # Remove rows with missing values
        feature_data = feature_data.dropna()
        
        if feature_data.empty:
            return {'error': 'No valid data after preprocessing'}
        
        # Scale features
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(scaled_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_stats(feature_data, cluster_labels)
        
        # Perform PCA for visualization
        pca_result = self._perform_pca(scaled_data, cluster_labels)
        
        return {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'inertia': kmeans.inertia_,
            'cluster_statistics': cluster_stats,
            'pca_analysis': pca_result,
            'silhouette_score': self._calculate_silhouette_score(scaled_data, cluster_labels)
        }
    
    def _find_optimal_clusters(self, data: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        max_clusters = min(max_clusters, len(data) - 1)
        
        if max_clusters < 2:
            return 1
        
        inertias = []
        k_range = range(1, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        if len(inertias) < 3:
            return 2
        
        # Calculate second derivative to find elbow
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_derivatives.append(inertias[i-1] - 2*inertias[i] + inertias[i+1])
        
        # Find the point with maximum second derivative
        elbow_index = np.argmax(second_derivatives) + 1
        return k_range[elbow_index]
    
    def _calculate_cluster_stats(self, data: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Calculate statistics for each cluster"""
        cluster_stats = {}
        
        for cluster_id in np.unique(labels):
            cluster_data = data[labels == cluster_id]
            
            stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100
            }
            
            # Calculate statistics for each feature
            for feature in data.columns:
                if data[feature].dtype in ['int64', 'float64']:
                    feature_data = cluster_data[feature]
                    stats[f'{feature}_mean'] = feature_data.mean()
                    stats[f'{feature}_std'] = feature_data.std()
                    stats[f'{feature}_min'] = feature_data.min()
                    stats[f'{feature}_max'] = feature_data.max()
                else:
                    # For categorical features, show most common value
                    stats[f'{feature}_mode'] = cluster_data[feature].mode().iloc[0] if not cluster_data[feature].empty else None
            
            cluster_stats[f'cluster_{cluster_id}'] = stats
        
        return cluster_stats
    
    def _perform_pca(self, data: np.ndarray, labels: np.ndarray) -> Dict:
        """Perform PCA for dimensionality reduction and visualization"""
        if data.shape[1] < 2:
            return {'error': 'Insufficient features for PCA'}
        
        # Perform PCA
        pca = PCA()
        pca_data = pca.fit_transform(data)
        
        # Calculate explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        return {
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'n_components_95': int(np.argmax(cumulative_variance >= 0.95)) + 1,
            'pca_coordinates': pca_data[:, :2].tolist(),  # First two components for visualization
            'feature_importance': pca.components_[:2].tolist()  # First two components
        }
    
    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality"""
        if len(np.unique(labels)) < 2:
            return 0.0
        
        # Simplified silhouette calculation
        n_samples = len(data)
        silhouette_scores = []
        
        for i in range(n_samples):
            # Calculate average distance to points in same cluster
            same_cluster = data[labels == labels[i]]
            if len(same_cluster) > 1:
                a = np.mean([np.linalg.norm(data[i] - point) for point in same_cluster if not np.array_equal(data[i], point)])
            else:
                a = 0
            
            # Calculate average distance to points in nearest cluster
            b = float('inf')
            for cluster_id in np.unique(labels):
                if cluster_id != labels[i]:
                    other_cluster = data[labels == cluster_id]
                    if len(other_cluster) > 0:
                        avg_dist = np.mean([np.linalg.norm(data[i] - point) for point in other_cluster])
                        b = min(b, avg_dist)
            
            # Calculate silhouette score for this point
            if max(a, b) > 0:
                silhouette_scores.append((b - a) / max(a, b))
            else:
                silhouette_scores.append(0)
        
        return np.mean(silhouette_scores)


class PredictiveModeler:
    """Predictive modeling for benchmark performance"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
    
    def build_performance_model(self, data: pd.DataFrame, target_column: str, 
                               feature_columns: List[str]) -> Dict:
        """Build predictive model for performance metrics"""
        if data.empty or target_column not in data.columns:
            return {'error': 'Invalid data or target column'}
        
        # Prepare features and target
        features = data[feature_columns].copy()
        target = data[target_column].copy()
        
        # Handle categorical variables
        for col in feature_columns:
            if features[col].dtype == 'object':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                features[col] = self.label_encoders[col].fit_transform(features[col].astype(str))
        
        # Remove rows with missing values
        valid_indices = features.dropna().index.intersection(target.dropna().index)
        features = features.loc[valid_indices]
        target = target.loc[valid_indices]
        
        if len(features) < 10:
            return {'error': 'Insufficient data for modeling'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for model_name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # Feature importance (for tree-based models)
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                for i, feature in enumerate(feature_columns):
                    feature_importance[feature] = model.feature_importances_[i]
            elif hasattr(model, 'coef_'):
                for i, feature in enumerate(feature_columns):
                    feature_importance[feature] = abs(model.coef_[i])
            
            results[model_name] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'feature_importance': feature_importance,
                'overfitting_score': abs(train_r2 - test_r2)
            }
            
            # Store model and scaler
            self.models[f'{target_column}_{model_name}'] = model
            self.scalers[f'{target_column}_{model_name}'] = scaler
        
        # Find best model
        best_model = max(results.keys(), key=lambda k: results[k]['test_r2'])
        
        return {
            'target_column': target_column,
            'feature_columns': feature_columns,
            'model_results': results,
            'best_model': best_model,
            'data_size': len(features),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def predict_performance(self, model_key: str, features: Dict) -> Dict:
        """Make performance predictions using trained model"""
        if model_key not in self.models:
            return {'error': f'Model {model_key} not found'}
        
        model = self.models[model_key]
        scaler = self.scalers[model_key]
        
        # Prepare features
        feature_array = []
        for feature_name in features.keys():
            value = features[feature_name]
            
            # Handle categorical encoding
            if feature_name in self.label_encoders:
                try:
                    value = self.label_encoders[feature_name].transform([str(value)])[0]
                except:
                    value = 0  # Unknown category
            
            feature_array.append(value)
        
        # Scale features
        feature_array = np.array(feature_array).reshape(1, -1)
        scaled_features = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        # Calculate prediction interval (simplified)
        if hasattr(model, 'predict') and len(scaled_features[0]) > 1:
            # Use model's training error as approximation for prediction uncertainty
            training_error = getattr(model, '_training_error', 0.1)
            prediction_interval = (prediction - 1.96 * training_error, 
                                 prediction + 1.96 * training_error)
        else:
            prediction_interval = (prediction * 0.9, prediction * 1.1)
        
        return {
            'prediction': prediction,
            'prediction_interval': prediction_interval,
            'model_used': model_key
        }


class AdvancedAnalyticsEngine:
    """Main analytics engine combining all analysis components"""
    
    def __init__(self, output_dir: str = "analytics_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.statistical_analyzer = StatisticalAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.cluster_analyzer = ClusterAnalyzer()
        self.predictive_modeler = PredictiveModeler()
        
        print(f"Advanced Analytics Engine initialized")
        print(f"Output directory: {self.output_dir}")
    
    def load_benchmark_data(self, data_dir: str) -> pd.DataFrame:
        """Load and consolidate benchmark data"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"Data directory {data_path} does not exist")
            return pd.DataFrame()
        
        # Load all JSON result files
        all_data = []
        
        for json_file in data_path.glob("**/*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Flatten nested structures
                flattened = self._flatten_dict(data)
                all_data.append(flattened)
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        if not all_data:
            print("No valid data files found")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Clean and standardize data
        df = self._clean_dataframe(df)
        
        print(f"Loaded {len(df)} benchmark records")
        return df
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary"""
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                # Handle numeric lists by taking mean
                items.append((new_key, np.mean(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize DataFrame"""
        # Convert timestamp columns
        timestamp_columns = [col for col in df.columns if 'timestamp' in col.lower() or 'time' in col.lower()]
        for col in timestamp_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['latency', 'memory', 'energy', 'power', 'accuracy', 'throughput'])]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        return df
    
    def comprehensive_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform comprehensive analysis on benchmark data"""
        if data.empty:
            return {'error': 'No data provided for analysis'}
        
        print("Starting comprehensive analysis...")
        
        analysis_results = {
            'data_summary': self._analyze_data_summary(data),
            'statistical_analysis': self._perform_statistical_analysis(data),
            'trend_analysis': self._perform_trend_analysis(data),
            'clustering_analysis': self._perform_clustering_analysis(data),
            'predictive_modeling': self._perform_predictive_modeling(data),
            'comparative_analysis': self._perform_comparative_analysis(data),
            'recommendations': self._generate_recommendations(data)
        }
        
        # Save results
        self._save_analysis_results(analysis_results)
        
        print("Comprehensive analysis completed")
        return analysis_results
    
    def _analyze_data_summary(self, data: pd.DataFrame) -> Dict:
        """Analyze data summary and quality"""
        summary = {
            'total_records': len(data),
            'total_features': len(data.columns),
            'missing_data_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'duplicate_records': data.duplicated().sum(),
            'data_types': data.dtypes.value_counts().to_dict()
        }
        
        # Analyze key metrics availability
        key_metrics = ['latency', 'memory', 'energy', 'accuracy', 'throughput']
        metric_availability = {}
        
        for metric in key_metrics:
            metric_columns = [col for col in data.columns if metric in col.lower()]
            if metric_columns:
                metric_data = data[metric_columns].dropna()
                metric_availability[metric] = {
                    'columns': metric_columns,
                    'available_records': len(metric_data),
                    'coverage_percentage': (len(metric_data) / len(data)) * 100
                }
        
        summary['metric_availability'] = metric_availability
        
        return summary
    
    def _perform_statistical_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform statistical analysis on numeric columns"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        statistical_results = {}
        
        for column in numeric_columns:
            column_data = data[column].dropna().tolist()
            
            if len(column_data) > 2:
                # Descriptive statistics
                desc_stats = self.statistical_analyzer.descriptive_statistics(column_data)
                
                # Normality test
                normality = self.statistical_analyzer.normality_test(column_data)
                
                statistical_results[column] = {
                    'descriptive_statistics': desc_stats,
                    'normality_test': normality
                }
        
        # Correlation analysis between key metrics
        correlation_results = {}
        key_numeric_columns = [col for col in numeric_columns 
                              if any(keyword in col.lower() 
                                   for keyword in ['latency', 'memory', 'energy', 'accuracy', 'throughput'])]
        
        for i, col1 in enumerate(key_numeric_columns):
            for col2 in key_numeric_columns[i+1:]:
                data1 = data[col1].dropna().tolist()
                data2 = data[col2].dropna().tolist()
                
                # Align data (same indices)
                common_indices = data[[col1, col2]].dropna().index
                if len(common_indices) > 2:
                    aligned_data1 = data.loc[common_indices, col1].tolist()
                    aligned_data2 = data.loc[common_indices, col2].tolist()
                    
                    correlation = self.statistical_analyzer.correlation_analysis(aligned_data1, aligned_data2)
                    correlation_results[f'{col1}_vs_{col2}'] = correlation
        
        return {
            'column_statistics': statistical_results,
            'correlations': correlation_results
        }
    
    def _perform_trend_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform trend analysis on time-series data"""
        timestamp_columns = [col for col in data.columns if 'timestamp' in col.lower()]
        
        if not timestamp_columns:
            return {'error': 'No timestamp columns found for trend analysis'}
        
        timestamp_col = timestamp_columns[0]
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        trend_results = {}
        
        for column in numeric_columns:
            # Get data with valid timestamps and values
            valid_data = data[[timestamp_col, column]].dropna()
            
            if len(valid_data) > 3:
                timestamps = valid_data[timestamp_col].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
                values = valid_data[column].tolist()
                
                trend_analysis = self.trend_analyzer.detect_trends(timestamps, values)
                trend_results[column] = trend_analysis
        
        return trend_results
    
    def _perform_clustering_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform clustering analysis"""
        # Select features for clustering
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = ['model_name', 'precision', 'benchmark_type']
        
        # Filter to existing columns
        available_categorical = [col for col in categorical_columns if col in data.columns]
        clustering_features = numeric_columns + available_categorical
        
        if len(clustering_features) < 2:
            return {'error': 'Insufficient features for clustering analysis'}
        
        # Remove columns with too many missing values
        valid_features = []
        for feature in clustering_features:
            if data[feature].notna().sum() / len(data) > 0.5:  # At least 50% valid data
                valid_features.append(feature)
        
        if len(valid_features) < 2:
            return {'error': 'Insufficient valid features for clustering'}
        
        clustering_result = self.cluster_analyzer.perform_clustering(data, valid_features)
        
        return clustering_result
    
    def _perform_predictive_modeling(self, data: pd.DataFrame) -> Dict:
        """Build predictive models for key performance metrics"""
        # Define target variables and features
        target_metrics = []
        feature_columns = []
        
        # Find target metrics
        for metric in ['latency_mean', 'memory_peak_mb', 'energy_joules', 'accuracy_score']:
            matching_columns = [col for col in data.columns if metric in col]
            if matching_columns:
                target_metrics.append(matching_columns[0])
        
        # Find feature columns
        categorical_features = ['model_name', 'precision', 'benchmark_type']
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for feature in categorical_features + numeric_features:
            if feature in data.columns and feature not in target_metrics:
                feature_columns.append(feature)
        
        if not target_metrics or len(feature_columns) < 2:
            return {'error': 'Insufficient data for predictive modeling'}
        
        modeling_results = {}
        
        for target in target_metrics:
            # Use a subset of features to avoid overfitting
            selected_features = feature_columns[:min(10, len(feature_columns))]
            
            model_result = self.predictive_modeler.build_performance_model(
                data, target, selected_features
            )
            
            modeling_results[target] = model_result
        
        return modeling_results
    
    def _perform_comparative_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform comparative analysis across different dimensions"""
        comparative_results = {}
        
        # Compare by model
        if 'model_name' in data.columns:
            model_comparison = self._compare_by_category(data, 'model_name')
            comparative_results['by_model'] = model_comparison
        
        # Compare by precision
        if 'precision' in data.columns:
            precision_comparison = self._compare_by_category(data, 'precision')
            comparative_results['by_precision'] = precision_comparison
        
        # Compare by benchmark type
        if 'benchmark_type' in data.columns:
            benchmark_comparison = self._compare_by_category(data, 'benchmark_type')
            comparative_results['by_benchmark_type'] = benchmark_comparison
        
        return comparative_results
    
    def _compare_by_category(self, data: pd.DataFrame, category_column: str) -> Dict:
        """Compare performance metrics across categories"""
        categories = data[category_column].unique()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        comparison_results = {}
        
        for metric in numeric_columns:
            metric_data = {}
            
            for category in categories:
                category_data = data[data[category_column] == category][metric].dropna().tolist()
                if len(category_data) > 0:
                    metric_data[category] = category_data
            
            if len(metric_data) > 1:
                # Perform statistical comparisons
                category_names = list(metric_data.keys())
                comparison_results[metric] = {}
                
                # Pairwise comparisons
                for i, cat1 in enumerate(category_names):
                    for cat2 in category_names[i+1:]:
                        comparison_key = f'{cat1}_vs_{cat2}'
                        
                        hypothesis_result = self.statistical_analyzer.hypothesis_testing(
                            metric_data[cat1], metric_data[cat2]
                        )
                        
                        comparison_results[metric][comparison_key] = hypothesis_result
        
        return comparison_results
    
    def _generate_recommendations(self, data: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Data quality recommendations
        missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if missing_percentage > 20:
            recommendations.append(
                f"Data quality concern: {missing_percentage:.1f}% of data is missing. "
                "Consider improving data collection processes."
            )
        
        # Performance recommendations
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if 'latency' in column.lower():
                latency_data = data[column].dropna()
                if len(latency_data) > 0:
                    high_latency_threshold = latency_data.quantile(0.9)
                    if latency_data.mean() > high_latency_threshold * 0.7:
                        recommendations.append(
                            f"High average latency detected in {column}. "
                            "Consider optimizing model inference or using more aggressive quantization."
                        )
            
            elif 'memory' in column.lower():
                memory_data = data[column].dropna()
                if len(memory_data) > 0 and memory_data.mean() > 8000:  # > 8GB
                    recommendations.append(
                        f"High memory usage detected in {column}. "
                        "Consider using smaller models or more aggressive quantization."
                    )
        
        # Precision recommendations
        if 'precision' in data.columns:
            precision_groups = data.groupby('precision')
            
            if 'accuracy_score' in data.columns:
                accuracy_by_precision = precision_groups['accuracy_score'].mean()
                
                if 'int4' in accuracy_by_precision.index and 'fp16' in accuracy_by_precision.index:
                    accuracy_drop = accuracy_by_precision['fp16'] - accuracy_by_precision['int4']
                    if accuracy_drop < 0.05:  # Less than 5% accuracy drop
                        recommendations.append(
                            "INT4 quantization shows minimal accuracy loss. "
                            "Consider using INT4 for production deployments to save memory and improve speed."
                        )
        
        # Model recommendations
        if 'model_name' in data.columns and len(data['model_name'].unique()) > 1:
            model_performance = {}
            
            for model in data['model_name'].unique():
                model_data = data[data['model_name'] == model]
                
                # Calculate composite performance score
                score = 0
                factors = 0
                
                if 'latency_mean' in data.columns:
                    latency = model_data['latency_mean'].mean()
                    if not np.isnan(latency):
                        score += 1 / (latency + 0.001)  # Lower latency is better
                        factors += 1
                
                if 'accuracy_score' in data.columns:
                    accuracy = model_data['accuracy_score'].mean()
                    if not np.isnan(accuracy):
                        score += accuracy  # Higher accuracy is better
                        factors += 1
                
                if factors > 0:
                    model_performance[model] = score / factors
            
            if model_performance:
                best_model = max(model_performance.keys(), key=lambda k: model_performance[k])
                recommendations.append(
                    f"Based on performance analysis, {best_model} shows the best "
                    "balance of speed and accuracy for your workload."
                )
        
        return recommendations
    
    def _save_analysis_results(self, results: Dict):
        """Save analysis results to files"""
        # Save main results
        results_file = self.output_dir / "comprehensive_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary report
        summary_file = self.output_dir / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("ADVANCED ANALYTICS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Data summary
            if 'data_summary' in results:
                f.write("DATA SUMMARY:\n")
                summary = results['data_summary']
                f.write(f"  Total Records: {summary.get('total_records', 0)}\n")
                f.write(f"  Total Features: {summary.get('total_features', 0)}\n")
                f.write(f"  Missing Data: {summary.get('missing_data_percentage', 0):.1f}%\n\n")
            
            # Key findings
            if 'recommendations' in results:
                f.write("KEY RECOMMENDATIONS:\n")
                for i, rec in enumerate(results['recommendations'], 1):
                    f.write(f"  {i}. {rec}\n")
                f.write("\n")
            
            # Statistical insights
            if 'statistical_analysis' in results and 'correlations' in results['statistical_analysis']:
                f.write("CORRELATION INSIGHTS:\n")
                correlations = results['statistical_analysis']['correlations']
                for pair, corr_data in correlations.items():
                    if 'pearson' in corr_data:
                        pearson = corr_data['pearson']
                        if pearson.get('significant', False):
                            f.write(f"  {pair}: {pearson['correlation']:.3f} ({pearson['strength']})\n")
        
        print(f"Analysis results saved to {self.output_dir}")


def main():
    """Main analytics demo"""
    analytics_engine = AdvancedAnalyticsEngine()
    
    # Generate sample data for demonstration
    sample_data = generate_sample_benchmark_data()
    
    # Perform comprehensive analysis
    results = analytics_engine.comprehensive_analysis(sample_data)
    
    print("\nAnalysis completed!")
    print(f"Results saved to: {analytics_engine.output_dir}")
    
    # Print key insights
    if 'recommendations' in results:
        print(f"\nKey Recommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")


def generate_sample_benchmark_data() -> pd.DataFrame:
    """Generate sample benchmark data for demonstration"""
    np.random.seed(42)
    
    models = ['model_small', 'model_medium', 'model_large']
    precisions = ['fp16', 'int8', 'int4']
    benchmark_types = ['latency', 'memory', 'energy', 'accuracy']
    
    data = []
    
    for i in range(200):
        model = np.random.choice(models)
        precision = np.random.choice(precisions)
        benchmark_type = np.random.choice(benchmark_types)
        
        # Simulate realistic relationships
        model_factor = {'model_small': 0.5, 'model_medium': 1.0, 'model_large': 2.0}[model]
        precision_factor = {'fp16': 1.0, 'int8': 0.7, 'int4': 0.5}[precision]
        
        record = {
            'model_name': model,
            'precision': precision,
            'benchmark_type': benchmark_type,
            'timestamp': (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat(),
            'latency_mean': np.random.normal(0.1 * model_factor / precision_factor, 0.02),
            'memory_peak_mb': np.random.normal(1000 * model_factor * precision_factor, 200),
            'energy_joules': np.random.normal(50 * model_factor / precision_factor, 10),
            'accuracy_score': np.random.normal(0.8 - (1 - precision_factor) * 0.1, 0.05),
            'throughput_tokens_per_sec': np.random.normal(100 / model_factor * precision_factor, 20)
        }
        
        # Ensure positive values
        for key in ['latency_mean', 'memory_peak_mb', 'energy_joules', 'throughput_tokens_per_sec']:
            record[key] = max(0.001, record[key])
        
        record['accuracy_score'] = max(0.0, min(1.0, record['accuracy_score']))
        
        data.append(record)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()