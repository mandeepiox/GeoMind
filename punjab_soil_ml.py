import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = '/Users/demo/ptu/testfile2.csv'
RANDOM_STATE = 57

class ImprovedMultiPropertyPredictor:
    """
    Enhanced ML model with data quality checks and feature engineering
    """
    
    def __init__(self):
        self.scalers = {}
        self.models = {}
        self.best_models = {}
        self.feature_columns = None
        self.target_columns = []
        self.poly_features = {}  # Store polynomial features per property
        
    def load_data(self, filepath):
        """Load and clean the dataset"""
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        
        try:
            self.data = pd.read_csv(filepath)
            
            # Clean unnamed/empty columns
            empty_cols = self.data.columns[self.data.isnull().all()].tolist()
            unnamed_cols = [col for col in self.data.columns if 'Unnamed' in str(col)]
            cols_to_drop = list(set(empty_cols + unnamed_cols))
            
            if cols_to_drop:
                self.data = self.data.drop(columns=cols_to_drop)
            
            self.data = self.data.dropna(how='all')
            
            print(f"✓ Loaded: {len(self.data)} rows, {len(self.data.columns)} columns")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        
        print("\nFirst 5 rows:")
        print(self.data.head())
        print("\nColumn names:")
        print(self.data.columns.tolist())
        
        return self.data
    
    def identify_columns(self):
        """Identify feature and target columns"""
        print("\n" + "="*70)
        print("IDENTIFYING COLUMNS")
        print("="*70)
        
        all_cols = self.data.columns.tolist()
        
        # Identify coordinate columns - LATITUDE FIRST, then LONGITUDE
        lat_patterns = ['latitude', 'lat']
        lon_patterns = ['longitude', 'lon', 'long']
        
        feature_cols = []
        lat_col = None
        lon_col = None
        
        # Find latitude column first
        for col in all_cols:
            if any(pattern in col.lower() for pattern in lat_patterns):
                lat_col = col
                break
        
        # Find longitude column
        for col in all_cols:
            if any(pattern in col.lower() for pattern in lon_patterns):
                lon_col = col
                break
        
        # Add in order: latitude, longitude
        if lat_col:
            feature_cols.append(lat_col)
        if lon_col:
            feature_cols.append(lon_col)
        
        # Identify depth column
        depth_patterns = ['depth', 'depth_m', 'depth (m)', 'depth(m)']
        for col in all_cols:
            if any(pattern in col.lower() for pattern in depth_patterns):
                if col not in feature_cols:
                    feature_cols.append(col)
        
        self.feature_columns = feature_cols
        print(f"\n✓ Feature columns (in order): {self.feature_columns}")
        
        # Identify target columns
        target_patterns = {
            'N-value': ['n_value', 'n value', 'n-value', 'spt', 'spt n', 'n_val'],
            'Bulk Density': ['bulk density', 'bulk_density', 'density', 'bd'],
            'Cohesion': ['cohesion', 'cohesive'],
            'Shear angle': ['shear angle', 'shear', 'phi', 'φ', 'friction angle', 'angle'],
            'Gravel': ['gravel', 'gravel %', 'gravel(%)'],
            'Sand': ['sand', 'sand %', 'sand(%)'],
            'Silt & Clay': ['silt', 'clay', 'silt & clay', 'silt and clay', 'silt+clay']
        }
        
        target_cols = []
        for col in all_cols:
            if col not in feature_cols:
                for target_name, patterns in target_patterns.items():
                    if any(pattern in col.lower() for pattern in patterns):
                        target_cols.append(col)
                        break
        
        self.target_columns = target_cols
        print(f"✓ Target columns: {self.target_columns}")
        
        return len(feature_cols) > 0 and len(target_cols) > 0
    
    def diagnose_data_quality(self):
        """Diagnose data quality issues for each property"""
        print("\n" + "="*70)
        print("DATA QUALITY DIAGNOSIS")
        print("="*70)
        
        self.data_quality = {}
        
        for target in self.target_columns:
            print(f"\n📊 Analyzing: {target}")
            print("-" * 50)
            
            # Get clean data
            data_clean = self.data.dropna(subset=[target] + self.feature_columns)
            
            if len(data_clean) < 10:
                print(f"  ❌ Insufficient data: only {len(data_clean)} rows")
                continue
            
            y = data_clean[target]
            X = data_clean[self.feature_columns]
            
            # Basic statistics
            print(f"  Data points: {len(y)}")
            print(f"  Range: [{y.min():.2f}, {y.max():.2f}]")
            print(f"  Mean: {y.mean():.2f}, Std: {y.std():.2f}")
            print(f"  Missing: {self.data[target].isnull().sum()} / {len(self.data)}")
            
            # Check for outliers (values beyond 3 std)
            outlier_threshold = 3
            outliers = np.abs((y - y.mean()) / y.std()) > outlier_threshold
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                print(f"  ⚠️  Outliers detected: {n_outliers} points ({100*n_outliers/len(y):.1f}%)")
                print(f"      Outlier values: {y[outliers].values[:5]}...")
            else:
                print(f"  ✓ No extreme outliers detected")
            
            # Check correlation with features
            print(f"\n  Correlation with features:")
            for feat in self.feature_columns:
                corr = data_clean[feat].corr(y)
                strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
                print(f"    {feat}: {corr:.3f} ({strength})")
            
            # Check data variability
            cv = (y.std() / y.mean()) * 100 if y.mean() != 0 else 0
            print(f"  Coefficient of Variation: {cv:.1f}%")
            
            if cv < 10:
                print(f"  ⚠️  Low variability - data might be too uniform")
            elif cv > 100:
                print(f"  ⚠️  High variability - data very scattered")
            
            # Store diagnosis
            self.data_quality[target] = {
                'n_samples': len(y),
                'n_outliers': n_outliers,
                'cv': cv,
                'correlations': {feat: data_clean[feat].corr(y) for feat in self.feature_columns}
            }
    
    def clean_outliers(self, target, threshold=3):
        """Remove outliers from a specific target"""
        data_clean = self.data.dropna(subset=[target] + self.feature_columns)
        y = data_clean[target]
        
        # Remove outliers beyond threshold standard deviations
        z_scores = np.abs((y - y.mean()) / y.std())
        mask = z_scores <= threshold
        
        cleaned_data = data_clean[mask]
        removed = len(data_clean) - len(cleaned_data)
        
        if removed > 0:
            print(f"  🧹 Removed {removed} outliers from {target}")
        
        return cleaned_data
    
    def preprocess_and_train(self, use_polynomial=True, remove_outliers=True):
        """Train models with enhanced features"""
        print("\n" + "="*70)
        print("TRAINING ENHANCED MODELS")
        print("="*70)
        
        self.results_summary = {}
        
        for target in self.target_columns:
            print(f"\n{'='*70}")
            print(f"TRAINING: {target}")
            print(f"{'='*70}")
            
            # Get clean data
            if remove_outliers:
                data_clean = self.clean_outliers(target, threshold=3)
            else:
                data_clean = self.data.dropna(subset=[target] + self.feature_columns)
            
            if len(data_clean) < 10:
                print(f"⚠️  Skipping {target} - insufficient data")
                continue
            
            X = data_clean[self.feature_columns].copy()
            y = data_clean[target].copy()
            
            print(f"Training samples: {len(X)}")
            print(f"Value range: [{y.min():.2f}, {y.max():.2f}]")
            
            # Feature engineering - add polynomial features for weak correlations
            max_corr = max([abs(X[feat].corr(y)) for feat in self.feature_columns])
            
            if use_polynomial and max_corr < 0.6:
                print(f"  🔧 Adding polynomial features (max correlation: {max_corr:.3f})")
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(X)
                X = pd.DataFrame(X_poly, index=X.index)
                self.poly_features[target] = poly
            else:
                print(f"  ✓ Using original features (max correlation: {max_corr:.3f})")
                self.poly_features[target] = None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[target] = scaler

            # Save test points for validation
            print(f"  ✓ Saving test points for {target}...")
            test_data_for_target = data_clean.loc[X_test.index]
            test_data_for_target.to_csv(f'test_set_{target}.csv', index=False)
            
            # Train multiple models
            models_to_try = {
                'Random Forest': RandomForestRegressor(
                    n_estimators=200, 
                    max_depth=10,
                    min_samples_split=5,
                    random_state=RANDOM_STATE
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=RANDOM_STATE
                ),
                'Ridge Regression': Ridge(alpha=1.0),
                'Linear Regression': LinearRegression()
            }
            
            best_score = -np.inf
            best_model = None
            best_model_name = None
            results = {}
            
            for name, model in models_to_try.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    test_r2 = r2_score(y_test, y_pred_test)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(
                        model, X_train_scaled, y_train, 
                        cv=min(5, len(X_train)//2), 
                        scoring='r2'
                    )
                    
                    results[name] = {
                        'test_r2': test_r2,
                        'test_rmse': test_rmse,
                        'test_mae': test_mae,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    print(f"  {name}:")
                    print(f"    R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
                    
                    if test_r2 > best_score:
                        best_score = test_r2
                        best_model = model
                        best_model_name = name
                        
                except Exception as e:
                    print(f"  ❌ {name} failed: {e}")
            
            if best_model is None:
                print(f"  ❌ All models failed for {target}")
                continue
            
            self.best_models[target] = best_model
            self.models[target] = models_to_try
            self.results_summary[target] = {
                'best_model': best_model_name,
                'best_r2': best_score,
                'results': results,
                'X_test': X_test_scaled,
                'y_test': y_test,
                'n_features': X.shape[1]
            }
            
            print(f"\n  ✓ Best: {best_model_name} (R² = {best_score:.4f})")
            
            # Interpretation with recommendations
            if best_score > 0.8:
                print(f"  ✓✓ EXCELLENT - Reliable predictions")
            elif best_score > 0.6:
                print(f"  ✓ GOOD - Usable predictions")
            elif best_score > 0.3:
                print(f"  ⚠️  WEAK - Use with caution, high uncertainty")
            else:
                print(f"  ❌ POOR - Predictions unreliable")
                print(f"     Recommendation: {target} may not correlate well with location/depth")
                print(f"     Consider: collecting more data or using different features")
    
    def plot_results(self):
        """Visualize model performance"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        if not self.results_summary:
            print("No results to plot")
            return
        
        # Model comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        properties = list(self.results_summary.keys())
        r2_scores = [self.results_summary[p]['best_r2'] for p in properties]
        best_models = [self.results_summary[p]['best_model'] for p in properties]
        
        # R² comparison
        colors = ['darkgreen' if r2 > 0.8 else 'green' if r2 > 0.6 else 'orange' if r2 > 0.3 else 'red' for r2 in r2_scores]
        bars = axes[0].barh(properties, r2_scores, color=colors, edgecolor='black', linewidth=1.5)
        axes[0].set_xlabel('R² Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Performance by Property', fontsize=14, fontweight='bold')
        axes[0].set_xlim([-0.5, 1])
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=1)
        axes[0].axvline(x=0.6, color='gray', linestyle='--', alpha=0.5, label='Good (0.6)')
        axes[0].axvline(x=0.8, color='gray', linestyle='--', alpha=0.7, label='Excellent (0.8)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        for i, (bar, r2, model) in enumerate(zip(bars, r2_scores, best_models)):
            x_pos = max(r2 + 0.02, 0.02)
            axes[0].text(x_pos, bar.get_y() + bar.get_height()/2, 
                        f'{r2:.3f}', va='center', fontsize=10, fontweight='bold')
        
        # Model type distribution
        model_counts = {}
        for model in best_models:
            model_counts[model] = model_counts.get(model, 0) + 1
        
        colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        axes[1].pie(model_counts.values(), labels=model_counts.keys(), autopct='%1.1f%%',
                   startangle=90, colors=colors_pie[:len(model_counts)])
        axes[1].set_title('Best Model Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison_enhanced.png', dpi=300, bbox_inches='tight')
        print("✓ Model comparison saved")
        plt.close()
        
        # Actual vs Predicted
        n_properties = len(properties)
        n_cols = 3
        n_rows = (n_properties + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.ravel()
        else:
            axes = axes.ravel()
        
        for idx, prop in enumerate(properties):
            if idx < len(axes):
                y_test = self.results_summary[prop]['y_test']
                X_test = self.results_summary[prop]['X_test']
                y_pred = self.best_models[prop].predict(X_test)
                r2 = self.results_summary[prop]['best_r2']
                
                axes[idx].scatter(y_test, y_pred, alpha=0.6, edgecolor='black', s=50)
                axes[idx].plot([y_test.min(), y_test.max()], 
                              [y_test.min(), y_test.max()], 
                              'r--', lw=2, label='Perfect')
                axes[idx].set_xlabel('Actual', fontsize=10)
                axes[idx].set_ylabel('Predicted', fontsize=10)
                
                color = 'darkgreen' if r2 > 0.8 else 'green' if r2 > 0.6 else 'orange' if r2 > 0.3 else 'red'
                axes[idx].set_title(f'{prop}\nR² = {r2:.3f}', fontsize=11, fontweight='bold', color=color)
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        for idx in range(n_properties, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('actual_vs_predicted_enhanced.png', dpi=300, bbox_inches='tight')
        print("✓ Actual vs Predicted plots saved")
        plt.close()
    
    def predict_all_properties(self, latitude, longitude, depth=None):
        """Predict all properties with uncertainty estimates"""
        print(f"\n{'='*70}")
        print(f"PREDICTIONS FOR ALL PROPERTIES")
        print(f"{'='*70}")
        
        # Prepare input - LATITUDE FIRST, then LONGITUDE
        if depth is not None and len(self.feature_columns) == 3:
            input_data = np.array([[latitude, longitude, depth]])
            print(f"Location: ({latitude}, {longitude}) at {depth}m depth")
        else:
            input_data = np.array([[latitude, longitude]])
            print(f"Location: ({latitude}, {longitude})")
        
        predictions = {}
        
        print(f"\n{'Property':<20} {'Prediction':<12} {'R²':<8}")
        print("-" * 50)
        
        for target in self.target_columns:
            if target in self.best_models:
                try:
                    # Apply polynomial features if used
                    if self.poly_features.get(target) is not None:
                        input_transformed = self.poly_features[target].transform(input_data)
                    else:
                        input_transformed = input_data
                    
                    # Scale input
                    input_scaled = self.scalers[target].transform(input_transformed)
                    
                    # Predict
                    prediction = self.best_models[target].predict(input_scaled)[0]
                    predictions[target] = prediction
                    
                    r2 = self.results_summary[target]['best_r2']
                    
                    print(f"{target:<20} {prediction:>10.2f}    {r2:>6.3f}")
                    
                except Exception as e:
                    print(f"{target:<20} {'ERROR':<12} {'N/A':<8}")
        
        print("="*70)
        
        return predictions
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*70)
        print("FINAL ANALYSIS REPORT")
        print("="*70)
        
        print(f"\n📊 Models trained: {len(self.best_models)} / {len(self.target_columns)}")
        print(f"📍 Input features: {', '.join(self.feature_columns)}")
        
        # Categorize by performance
        excellent = [t for t in self.results_summary if self.results_summary[t]['best_r2'] > 0.8]
        good = [t for t in self.results_summary if 0.6 < self.results_summary[t]['best_r2'] <= 0.8]
        weak = [t for t in self.results_summary if 0.3 < self.results_summary[t]['best_r2'] <= 0.6]
        poor = [t for t in self.results_summary if self.results_summary[t]['best_r2'] <= 0.3]
        
        print(f"\n🎯 Performance Summary:")
        print(f"  ✓✓ Excellent (R² > 0.8): {len(excellent)}")
        for p in excellent:
            print(f"     • {p}: {self.results_summary[p]['best_r2']:.3f}")
        
        print(f"\n  ✓  Good (0.6 < R² ≤ 0.8): {len(good)}")
        for p in good:
            print(f"     • {p}: {self.results_summary[p]['best_r2']:.3f}")
        
        print(f"\n  ⚠️  Weak (0.3 < R² ≤ 0.6): {len(weak)}")
        for p in weak:
            print(f"     • {p}: {self.results_summary[p]['best_r2']:.3f}")
            print(f"       → Use predictions with caution, high uncertainty")
        
        print(f"\n  ❌ Poor (R² ≤ 0.3): {len(poor)}")
        for p in poor:
            print(f"     • {p}: {self.results_summary[p]['best_r2']:.3f}")
            print(f"       → Predictions unreliable - these properties likely don't")
            print(f"         correlate well with location/depth alone")
        
        if poor:
            print(f"\n💡 Recommendations for poor-performing properties:")
            print(f"   1. These may require additional features (soil type, geology, climate)")
            print(f"   2. Consider collecting more diverse data")
            print(f"   3. May need different prediction approach (clustering, etc.)")
        
        print("\n" + "="*70)


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("ENHANCED MULTI-PROPERTY SOIL PREDICTION")
    print("="*70)
    
    predictor = ImprovedMultiPropertyPredictor()
    
    try:
        # Load data
        data = predictor.load_data(DATA_PATH)
        if data is None:
            exit(1)
        
        # Identify columns
        if not predictor.identify_columns():
            print("Failed to identify columns")
            exit(1)
        
        # Diagnose data quality
        predictor.diagnose_data_quality()
        
        # Train enhanced models
        predictor.preprocess_and_train(use_polynomial=True, remove_outliers=True)
        
        # Plot results
        predictor.plot_results()
        
        # Generate report
        predictor.generate_report()
        
        # Sample predictions - LATITUDE FIRST, then LONGITUDE
        print("\n" + "="*70)
        print("SAMPLE PREDICTIONS")
        print("="*70)
        
        sample_coords = [
            (31.227389, 75.766764, 3.0),
            
        ]
        
        for lat, lon, depth in sample_coords:
            predictor.predict_all_properties(lat, lon, depth)
        
        print("\n✓ Analysis complete!")
        
        # Interactive mode
        print("\n" + "="*70)
        print("INTERACTIVE MODE")
        print("="*70)
        print("Enter coordinates to predict (or 'quit'):\n")
        
        while True:
            try:
                user_input = input("\nEnter (lat, lon, depth) or 'quit': ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                parts = user_input.split(',')
                if len(parts) != 3:
                    print("❌ Need 3 values: latitude, longitude, depth")
                    continue
                
                lat, lon, dep = float(parts[0]), float(parts[1]), float(parts[2])
                predictor.predict_all_properties(lat, lon, dep)
                
            except ValueError:
                print("❌ Enter valid numbers")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()