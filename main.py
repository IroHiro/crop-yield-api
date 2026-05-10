"""
TERRASIGHT PREDICTION API - WITH MUNICIPALITY AND SEASON FEATURES
Supports: Rice, Corn, and Coconut predictions
Rice/Corn expects: 16 season summary + 1 crop + 1 municipality + 1 season + 4 typhoon = 23 features
Coconut expects: 16 season summary + 1 municipality + optional lag feature
"""

import numpy as np
import pandas as pd
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("TERRASIGHT PREDICTION API - Starting up...")
print("=" * 70)

# ============================================
# LOAD RICE/CORN MODELS
# ============================================
print("\n📦 Loading Rice/Corn models...")

rf_model = None
crop_encoder = None
municipality_encoder = None
season_encoder = None
scaler = None

rf_path = os.path.join(BASE_DIR, 'random_forest_model.pkl')
if os.path.exists(rf_path):
    rf_model = joblib.load(rf_path)
    print(f"✅ Random Forest loaded - expects {rf_model.n_features_in_} features")
else:
    rf_model = None
    print(f"❌ random_forest_model.pkl not found")

crop_encoder_path = os.path.join(BASE_DIR, 'crop_encoder.pkl')
if os.path.exists(crop_encoder_path):
    crop_encoder = joblib.load(crop_encoder_path)
    print(f"✅ Crop encoder loaded: {crop_encoder.classes_}")
else:
    crop_encoder = None

municipality_encoder_path = os.path.join(BASE_DIR, 'municipality_encoder.pkl')
if os.path.exists(municipality_encoder_path):
    municipality_encoder = joblib.load(municipality_encoder_path)
    print(f"✅ Municipality encoder loaded: {municipality_encoder.classes_}")
else:
    municipality_encoder = None

season_encoder_path = os.path.join(BASE_DIR, 'season_encoder.pkl')
if os.path.exists(season_encoder_path):
    season_encoder = joblib.load(season_encoder_path)
    print(f"✅ Season encoder loaded: {season_encoder.classes_}")
else:
    season_encoder = None

scaler_path = os.path.join(BASE_DIR, 'feature_scaler.pkl')
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print(f"✅ Feature scaler loaded")
else:
    scaler = None
    print(f"⚠️ feature_scaler.pkl not found - predictions may be off")

feature_names_path = os.path.join(BASE_DIR, 'rf_feature_names.npy')
if os.path.exists(feature_names_path):
    feature_names = np.load(feature_names_path, allow_pickle=True)
    print(f"✅ Feature names loaded: {len(feature_names)} features")
else:
    feature_names = None

# ============================================
# LOAD COCONUT MODELS
# ============================================
print("\n📦 Loading Coconut models...")

coconut_model = None
coconut_mun_encoder = None
coconut_mun_baselines = None
coconut_scaler = None

coconut_model_path = os.path.join(BASE_DIR, 'coconut_improved_model.pkl')
if os.path.exists(coconut_model_path):
    coconut_model = joblib.load(coconut_model_path)
    print(f"✅ Coconut model loaded - expects {coconut_model.n_features_in_} features")
else:
    coconut_model = None
    print(f"⚠️ coconut_improved_model.pkl not found - Coconut predictions disabled")

coconut_mun_encoder_path = os.path.join(BASE_DIR, 'coconut_municipality_encoder.pkl')
if os.path.exists(coconut_mun_encoder_path):
    coconut_mun_encoder = joblib.load(coconut_mun_encoder_path)
    print(f"✅ Coconut municipality encoder loaded: {list(coconut_mun_encoder.classes_)}")
else:
    coconut_mun_encoder = None

coconut_baselines_path = os.path.join(BASE_DIR, 'coconut_mun_baselines.pkl')
if os.path.exists(coconut_baselines_path):
    coconut_mun_baselines = joblib.load(coconut_baselines_path)
    print(f"✅ Coconut baselines loaded")
    for mun, baseline in coconut_mun_baselines.items():
        print(f"   {mun}: {baseline/1e6:.2f}M nuts")
else:
    coconut_mun_baselines = None

coconut_scaler_path = os.path.join(BASE_DIR, 'coconut_scaler.pkl')
if os.path.exists(coconut_scaler_path):
    coconut_scaler = joblib.load(coconut_scaler_path)
    print(f"✅ Coconut scaler loaded")
else:
    coconut_scaler = None

print("=" * 70)

# ============================================
# HELPER FUNCTIONS
# ============================================

def process_weekly_sequence(raw_sequence):
    """
    Convert 4 weeks of weather data into 16 season summary features
    (matching the LSTM output from training)
    """
    # Calculate key statistics across the 4 weeks
    ndvi_vals = [w[0] for w in raw_sequence]
    evi_vals = [w[1] for w in raw_sequence]
    temp_vals = [w[4] for w in raw_sequence]
    temp_max_vals = [w[2] for w in raw_sequence]
    temp_min_vals = [w[3] for w in raw_sequence]
    rain_vals = [w[5] for w in raw_sequence]
    hum_vals = [w[6] for w in raw_sequence]
    solar_vals = [w[7] for w in raw_sequence]
    wind_vals = [w[8] for w in raw_sequence]
    
    # Create 16 summary features (matching LSTM output)
    features = [
        # NDVI (3)
        np.mean(ndvi_vals), np.max(ndvi_vals), np.min(ndvi_vals),
        # EVI (3)
        np.mean(evi_vals), np.max(evi_vals), np.min(evi_vals),
        # Temperature (3)
        np.mean(temp_vals), np.max(temp_vals), np.min(temp_vals),
        # Rainfall (2)
        np.sum(rain_vals), np.mean(rain_vals),
        # Humidity (2)
        np.mean(hum_vals), np.max(hum_vals),
        # Solar (2)
        np.mean(solar_vals), np.max(solar_vals),
        # Wind (1)
        np.mean(wind_vals)
    ]
    
    return np.array(features)

def process_coconut_sequence(raw_sequence):
    """
    Convert weekly weather sequence to season summary features for Coconut
    Returns 16 features matching training data
    """
    if not raw_sequence or len(raw_sequence) == 0:
        return np.zeros(16)
    
    weeks = raw_sequence[:4]
    
    ndvi_vals = [w[0] for w in weeks]
    evi_vals = [w[1] for w in weeks]
    temp_vals = [w[4] for w in weeks]
    rain_vals = [w[5] for w in weeks]
    hum_vals = [w[6] for w in weeks]
    solar_vals = [w[7] if len(w) > 7 else 0 for w in weeks]
    wind_vals = [w[8] if len(w) > 8 else 0 for w in weeks]
    
    features = [
        np.mean(ndvi_vals), np.max(ndvi_vals), np.min(ndvi_vals),
        np.mean(evi_vals), np.max(evi_vals), np.min(evi_vals),
        np.mean(temp_vals), np.max(temp_vals), np.min(temp_vals),
        np.sum(rain_vals), np.mean(rain_vals),
        np.mean(hum_vals), np.max(hum_vals),
        np.mean(solar_vals), np.max(solar_vals),
        np.mean(wind_vals)
    ]
    return np.array(features)

def get_season_name(season_code):
    if season_code == 2:
        return 'Wet'
    elif season_code == 1:
        return 'Dry'
    return None

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data'}), 400
        
        # Get crop type from request
        crop = data.get('crop', '').lower()
        raw_sequence = data.get('raw_sequence')
        municipality = data.get('municipality')
        
        if not raw_sequence:
            return jsonify({'error': 'Missing raw_sequence'}), 400
        if not municipality:
            return jsonify({'error': 'Missing municipality'}), 400
        
        # ============================================
        # COCONUT PREDICTION
        # ============================================
        if crop == 'coconut':
            if coconut_model is None:
                return jsonify({'error': 'Coconut model not loaded. Please check server configuration.'}), 500
            
            previous_yield = data.get('previous_yield', 0)
            
            print(f"\n{'='*60}")
            print(f"📥 COCONUT REQUEST: {municipality}")
            print(f"{'='*60}")
            
            # Process weather sequence to 16 features
            season_features = process_coconut_sequence(raw_sequence)
            print(f"Season features (16): {[round(x, 4) for x in season_features[:5]]}...")
            
            # Encode municipality
            if coconut_mun_encoder:
                if municipality in coconut_mun_encoder.classes_:
                    muni_code = coconut_mun_encoder.transform([municipality])[0]
                else:
                    # Try case-insensitive match
                    found = False
                    for cls in coconut_mun_encoder.classes_:
                        if cls.lower() == municipality.lower():
                            muni_code = coconut_mun_encoder.transform([cls])[0]
                            municipality = cls
                            found = True
                            break
                    if not found:
                        return jsonify({'error': f'Municipality "{municipality}" not recognized for coconut. Supported: {list(coconut_mun_encoder.classes_)}'}), 400
            else:
                return jsonify({'error': 'Municipality encoder not loaded'}), 500
            
            print(f"Municipality code: {muni_code}")
            
            # Build feature vector
            feature_list = list(season_features) + [muni_code]
            
            # Add lag feature if model expects 22 features
            expected_features = coconut_model.n_features_in_
            if expected_features == 22:
                feature_list.append(previous_yield)
                print(f"Added previous yield: {previous_yield:,.0f} nuts")
            
            print(f"Total features: {len(feature_list)} (model expects {expected_features})")
            
            # Pad or trim if needed
            if len(feature_list) < expected_features:
                feature_list.extend([0] * (expected_features - len(feature_list)))
                print(f"Padded to {expected_features} features")
            elif len(feature_list) > expected_features:
                feature_list = feature_list[:expected_features]
                print(f"Trimmed to {expected_features} features")
            
            # Apply scaler if available
            features_array = np.array(feature_list).reshape(1, -1)
            if coconut_scaler:
                features_array = coconut_scaler.transform(features_array)
                print("✅ Applied feature scaling")
            
            # Predict
            prediction = coconut_model.predict(features_array)[0]
            
            # Convert from ratio to absolute if baselines available
            if coconut_mun_baselines and municipality in coconut_mun_baselines:
                baseline = coconut_mun_baselines[municipality]
                predicted_yield = prediction * baseline
                print(f"Converted ratio {prediction:.3f} * baseline {baseline/1e6:.2f}M = {predicted_yield/1e6:.2f}M nuts")
            else:
                predicted_yield = prediction
            
            print(f"\n✅ COCONUT PREDICTION: {predicted_yield:,.0f} nuts ({predicted_yield/1e6:.2f}M nuts)")
            print(f"{'='*60}\n")
            
            return jsonify({
                'success': True,
                'crop': 'coconut',
                'municipality': municipality,
                'predicted_yield_nuts': float(predicted_yield),
                'predicted_yield_millions': float(predicted_yield / 1e6),
                'prediction_ratio': float(prediction),
                'features_used': len(feature_list)
            })
        
        # ============================================
        # RICE / CORN PREDICTION (ORIGINAL CODE - UNCHANGED)
        # ============================================
        elif crop in ['rice', 'corn']:
            if rf_model is None:
                return jsonify({'error': 'Rice/Corn model not loaded'}), 500
            
            crop_encoded = data.get('crop_encoded')
            season_code = data.get('season')
            
            # Typhoon parameters (optional, default to 0)
            typhoon_params = {
                'max_wind_kts': data.get('max_wind_kts', 0),
                'min_pres_mb': data.get('min_pres_mb', 1013),
                'duration_hrs': data.get('duration_hrs', 0),
                'risk_score': data.get('risk_score', 0)
            }
            
            if crop_encoded is None:
                return jsonify({'error': 'Missing crop_encoded'}), 400
            
            # Get 16 season summary features
            season_features = process_weekly_sequence(raw_sequence)
            crop_name = 'Rice' if crop_encoded == 1 else 'Corn'
            season_name = get_season_name(season_code)
            
            print(f"\n{'='*60}")
            print(f"📥 REQUEST: {municipality}, {crop_name}, {season_name}")
            print(f"{'='*60}")
            print(f"Season summary features (16): {[round(x, 4) for x in season_features]}")
            
            # Build COMPLETE feature vector matching training (23 features)
            feature_list = list(season_features)  # 16 features
            
            # Add crop (Rice=1, Corn=0)
            crop_value = crop_encoded
            feature_list.append(crop_value)
            print(f"Added crop '{crop_name}' = {crop_value}")
            
            # Add municipality (using encoder if available)
            if municipality_encoder and municipality in municipality_encoder.classes_:
                muni_value = municipality_encoder.transform([municipality])[0]
            else:
                # Fallback mapping
                muni_map = {'Ligao': 0, 'Malinao': 1, 'Oas': 2, 'Pioduran': 3, 'Polangui': 4}
                muni_value = muni_map.get(municipality, 0)
                print(f"⚠️ Using fallback mapping for municipality: {municipality} → {muni_value}")
            feature_list.append(muni_value)
            print(f"Added municipality '{municipality}' = {muni_value}")
            
            # Add season (Dry=0, Wet=1)
            season_value = 1 if season_name == 'Wet' else 0
            feature_list.append(season_value)
            print(f"Added season '{season_name}' = {season_value}")
            
            # Add typhoon parameters
            feature_list.append(typhoon_params['max_wind_kts'])
            feature_list.append(typhoon_params['min_pres_mb'])
            feature_list.append(typhoon_params['duration_hrs'])
            feature_list.append(typhoon_params['risk_score'])
            print(f"Added typhoon params: wind={typhoon_params['max_wind_kts']}, pressure={typhoon_params['min_pres_mb']}, duration={typhoon_params['duration_hrs']}, risk={typhoon_params['risk_score']}")
            
            # Check feature count
            expected = rf_model.n_features_in_ if rf_model else 23
            print(f"\n📊 Feature breakdown:")
            print(f"   Season summary features: 16")
            print(f"   Crop encoded: 1")
            print(f"   Municipality encoded: 1")
            print(f"   Season encoded: 1")
            print(f"   Typhoon parameters: 4")
            print(f"   Total: {len(feature_list)} features")
            print(f"   Model expects: {expected} features")
            
            if len(feature_list) != expected:
                print(f"⚠️ Feature count mismatch!")
                if len(feature_list) < expected:
                    feature_list.extend([0] * (expected - len(feature_list)))
                    print(f"   Padded with {expected - len(feature_list)} zeros")
                else:
                    feature_list = feature_list[:expected]
                    print(f"   Trimmed to {expected} features")
            
            # Apply scaler if available
            features_array = np.array(feature_list).reshape(1, -1)
            
            if scaler:
                features_array = scaler.transform(features_array)
                print(f"\n✅ Applied feature scaling")
            
            # Print final feature vector (truncated for readability)
            print(f"\n🔢 FINAL FEATURE VECTOR (first 5, last 5):")
            print(f"   {[round(x, 4) for x in features_array[0][:5]]} ... {[round(x, 4) for x in features_array[0][-5:]]}")
            
            # Predict
            predicted = rf_model.predict(features_array)[0]
            
            print(f"\n✅ PREDICTION RESULT: {predicted:.4f} tons/ha")
            print(f"{'='*60}\n")
            
            return jsonify({
                'success': True,
                'yield_tons': round(predicted, 4),
                'yield_kg': round(predicted * 1000, 0),
                'crop': crop_name,
                'municipality': municipality,
                'season': season_name,
                'features_used': len(feature_list),
                'model_expects': expected,
                'status': 'success'
            })
        
        else:
            return jsonify({'error': f'Crop "{crop}" not supported. Use: rice, corn, or coconut'}), 400
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information including expected features and encoders"""
    info = {
        'rice_corn': {
            'rf_features_expected': rf_model.n_features_in_ if rf_model else None,
            'crop_encoder_classes': crop_encoder.classes_.tolist() if crop_encoder else None,
            'municipality_encoder_classes': municipality_encoder.classes_.tolist() if municipality_encoder else None,
            'season_encoder_classes': season_encoder.classes_.tolist() if season_encoder else None,
            'scaler_available': scaler is not None
        },
        'coconut': {
            'model_loaded': coconut_model is not None,
            'features_expected': coconut_model.n_features_in_ if coconut_model else None,
            'municipalities': list(coconut_mun_encoder.classes_) if coconut_mun_encoder else [],
            'baselines_available': coconut_mun_baselines is not None,
            'scaler_available': coconut_scaler is not None
        }
    }
    
    if feature_names is not None:
        info['rice_corn']['feature_names'] = feature_names.tolist()
    
    return jsonify(info)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models': {
            'rice_corn': rf_model is not None,
            'coconut': coconut_model is not None
        }
    })

@app.route('/supported_crops', methods=['GET'])
def supported_crops():
    return jsonify({
        'crops': ['rice', 'corn', 'coconut'],
        'rice_corn_municipalities': list(municipality_encoder.classes_) if municipality_encoder else [],
        'coconut_municipalities': list(coconut_mun_encoder.classes_) if coconut_mun_encoder else []
    })

@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    """Get feature importance from the Rice/Corn model"""
    if rf_model is None:
        return jsonify({'error': 'Rice/Corn model not loaded'}), 500
    
    importances = rf_model.feature_importances_.tolist()
    
    result = {
        'feature_importances': importances,
        'num_features': len(importances)
    }
    
    if feature_names is not None and len(feature_names) == len(importances):
        # Sort by importance
        sorted_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        result['feature_importance_sorted'] = [
            {'name': name, 'importance': imp} for name, imp in sorted_pairs
        ]
    
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
