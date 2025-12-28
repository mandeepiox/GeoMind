"""
Train the ML model and save it for the Flask backend
Run this script after training to save models for production use
"""

import pickle
import os
from punjab_soil_ml import ImprovedMultiPropertyPredictor

# Configuration
DATA_PATH = '/Users/demo/ptu/testfile2.csv'
MODEL_DIR = 'models'

def save_models(predictor, model_dir='models'):
    """Save trained models and state for Flask backend"""
    os.makedirs(model_dir, exist_ok=True)
    
    state = {
        'scalers': predictor.scalers,
        'models': predictor.best_models,
        'poly_features': predictor.poly_features,
        'feature_columns': predictor.feature_columns,
        'target_columns': predictor.target_columns,
        'results_summary': predictor.results_summary
    }
    
    with open(f'{model_dir}/predictor_state.pkl', 'wb') as f:
        pickle.dump(state, f)
    
    print(f"✓ Models saved to {model_dir}/predictor_state.pkl")
    print(f"✓ Saved {len(predictor.best_models)} models")
    print(f"✓ Target properties: {predictor.target_columns}")

def main():
    """Train model and save for production"""
    print("="*70)
    print("TRAINING AND SAVING MODELS FOR PRODUCTION")
    print("="*70)
    
    # Initialize predictor
    predictor = ImprovedMultiPropertyPredictor()
    
    # Load data
    print("\n📁 Loading data...")
    data = predictor.load_data(DATA_PATH)
    if data is None:
        print("❌ Failed to load data")
        return
    
    # Identify columns
    print("\n🔍 Identifying columns...")
    if not predictor.identify_columns():
        print("❌ Failed to identify columns")
        return
    
    # Diagnose data quality
    print("\n🔬 Diagnosing data quality...")
    predictor.diagnose_data_quality()
    
    # Train models
    print("\n🤖 Training models...")
    predictor.preprocess_and_train(use_polynomial=True, remove_outliers=True)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    predictor.plot_results()
    
    # Generate report
    predictor.generate_report()
    
    # Save models for Flask
    print("\n💾 Saving models for Flask backend...")
    save_models(predictor, MODEL_DIR)
    
    # Test prediction
    print("\n✅ Testing prediction with saved models...")
    test_pred = predictor.predict_all_properties(31.227389, 75.766764, 3.0)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Models saved in '{MODEL_DIR}/' directory")
    print(f"2. Start Flask backend: python app.py")
    print(f"3. Open frontend: index.html in browser")
    print(f"4. Make predictions through the web interface!")

if __name__ == "__main__":
    main()