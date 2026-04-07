from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and preprocessing components
model = None
scaler = None
label_encoders = None

def load_model_components():
    """Load the trained model and preprocessing components"""
    global model, scaler, label_encoders
    
    try:
        model_path = os.getenv('MODEL_PATH', 'models/fraud_model.pkl')
        scaler_path = os.getenv('SCALER_PATH', 'models/scaler.pkl')
        encoders_path = 'models/label_encoders.pkl'
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(encoders_path)
        
        logger.info("Model components loaded successfully")
        return True
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading model components: {e}")
        return False

def extract_features(transaction_data):
    """Extract features from transaction data for ML model"""
    
    # Parse timestamp
    timestamp = pd.to_datetime(transaction_data.get('timestamp', datetime.now()))
    
    # Extract time-based features
    hour_of_day = timestamp.hour
    day_of_week = timestamp.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # For demo purposes, generate some realistic values for frequency-based features
    # In a real system, these would be calculated from historical data
    np.random.seed(hash(transaction_data.get('senderUpiId', '')) % 1000)
    
    # Simulate sender transaction frequency (transactions per day)
    sender_frequency = np.random.exponential(5)
    
    # Simulate receiver transaction frequency
    receiver_frequency = np.random.exponential(3)
    
    # Simulate location mismatch (1 if location is unusual for this user)
    location_mismatch = np.random.choice([0, 1], p=[0.9, 0.1])
    
    # Simulate device change (1 if device is new for this user)
    device_change = np.random.choice([0, 1], p=[0.85, 0.15])
    
    # Calculate amount deviation from user's average
    amount = float(transaction_data.get('amount', 0))
    amount_deviation = np.random.exponential(2)  # Deviation factor
    
    # Simulate time since last transaction (minutes)
    time_since_last = np.random.exponential(60)
    
    # Transaction type (default to p2p if not provided)
    transaction_type = transaction_data.get('transactionType', 'p2p')
    
    features = {
        'amount': amount,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'sender_frequency': sender_frequency,
        'receiver_frequency': receiver_frequency,
        'location_mismatch': location_mismatch,
        'device_change': device_change,
        'amount_deviation': amount_deviation,
        'time_since_last': time_since_last,
        'is_weekend': is_weekend,
        'transaction_type': transaction_type
    }
    
    return features

def preprocess_transaction(features):
    """Preprocess transaction features for prediction"""
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Encode categorical variables
    df['transaction_type_encoded'] = label_encoders['transaction_type'].transform(df['transaction_type'])
    
    # Select features in the correct order
    feature_columns = [
        'amount', 'hour_of_day', 'day_of_week', 'sender_frequency',
        'receiver_frequency', 'location_mismatch', 'device_change',
        'amount_deviation', 'time_since_last', 'is_weekend',
        'transaction_type_encoded'
    ]
    
    X = df[feature_columns]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled

def determine_risk_factors(features, fraud_score):
    """Determine risk factors based on features and fraud score"""
    
    risk_factors = []
    
    if features['amount'] > 50000:
        risk_factors.append('high_amount')
    
    if features['hour_of_day'] < 6 or features['hour_of_day'] > 22:
        risk_factors.append('unusual_time')
    
    if features['location_mismatch'] == 1:
        risk_factors.append('location_mismatch')
    
    if features['device_change'] == 1:
        risk_factors.append('new_device')
    
    if features['amount_deviation'] > 5:
        risk_factors.append('amount_anomaly')
    
    if features['time_since_last'] < 5:
        risk_factors.append('high_frequency')
    
    if features['is_weekend'] == 1 and features['amount'] > 20000:
        risk_factors.append('weekend_high_value')
    
    return risk_factors

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'UPI Fraud Detection ML API is running',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/fraud/check', methods=['POST'])
def check_fraud():
    """Check transaction for fraud"""
    
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'ML model is not available'
        }), 503
    
    try:
        # Validate input data
        required_fields = ['senderUpiId', 'receiverUpiId', 'amount', 'deviceId']
        
        for field in required_fields:
            if field not in request.json:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        transaction_data = request.json
        
        # Extract features
        features = extract_features(transaction_data)
        
        # Preprocess features
        X_scaled = preprocess_transaction(features)
        
        # Make prediction
        fraud_probability = model.predict_proba(X_scaled)[0, 1]
        fraud_score = round(fraud_probability * 100, 2)
        
        # Determine status
        if fraud_score >= 70:
            status = 'fraud'
        elif fraud_score >= 40:
            status = 'suspicious'
        else:
            status = 'safe'
        
        # Determine risk factors
        risk_factors = determine_risk_factors(features, fraud_score)
        
        # Log the prediction
        logger.info(f"Fraud check completed - Score: {fraud_score}%, Status: {status}")
        
        return jsonify({
            'fraudScore': fraud_score,
            'status': status,
            'riskFactors': risk_factors,
            'features': features,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error during fraud check: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Failed to process fraud check'
        }), 500

@app.route('/api/fraud/batch-check', methods=['POST'])
def batch_check_fraud():
    """Check multiple transactions for fraud"""
    
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'ML model is not available'
        }), 503
    
    try:
        transactions = request.json.get('transactions', [])
        
        if not transactions:
            return jsonify({
                'error': 'No transactions provided'
            }), 400
        
        results = []
        
        for transaction_data in transactions:
            try:
                # Extract features
                features = extract_features(transaction_data)
                
                # Preprocess features
                X_scaled = preprocess_transaction(features)
                
                # Make prediction
                fraud_probability = model.predict_proba(X_scaled)[0, 1]
                fraud_score = round(fraud_probability * 100, 2)
                
                # Determine status
                if fraud_score >= 70:
                    status = 'fraud'
                elif fraud_score >= 40:
                    status = 'suspicious'
                else:
                    status = 'safe'
                
                # Determine risk factors
                risk_factors = determine_risk_factors(features, fraud_score)
                
                results.append({
                    'transactionId': transaction_data.get('transactionId'),
                    'fraudScore': fraud_score,
                    'status': status,
                    'riskFactors': risk_factors
                })
                
            except Exception as e:
                logger.error(f"Error processing transaction: {e}")
                results.append({
                    'transactionId': transaction_data.get('transactionId'),
                    'error': 'Failed to process transaction'
                })
        
        return jsonify({
            'results': results,
            'processedAt': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error during batch fraud check: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Failed to process batch fraud check'
        }), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503
    
    try:
        feature_names = [
            'amount', 'hour_of_day', 'day_of_week', 'sender_frequency',
            'receiver_frequency', 'location_mismatch', 'device_change',
            'amount_deviation', 'time_since_last', 'is_weekend',
            'transaction_type_encoded'
        ]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return jsonify({
            'modelType': 'RandomForestClassifier',
            'features': feature_names,
            'featureImportance': feature_importance.to_dict('records'),
            'nFeatures': len(feature_names),
            'modelLoaded': True
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'error': 'Failed to get model information'
        }), 500
import os
if __name__ == '__main__':
    # Load model components
    if load_model_components():
        logger.info("Starting ML API server...")
        port = int(os.environ.get('PORT', 8000))
        app.run(host='0.0.0.0', port=port)
    else:
        logger.error("Failed to load model components. Please train the model first.")
        print("Please run 'python train_model.py' first to train and save the model.")
