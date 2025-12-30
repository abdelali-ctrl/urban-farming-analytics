import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from config import Config

class ModelPredictor:
    def __init__(self):
        self.config = Config()
        self.model = None
        self.pipeline = None
        self.scaler = StandardScaler()
        self.model_loaded = False
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load trained models"""
        try:
            # Try to load the pipeline first
            self.pipeline = joblib.load(self.config.PIPELINE_MODEL)
            self.model_loaded = True
            print("[OK] Pipeline model loaded successfully")
        except Exception as e:
            print(f"[WARN] Error loading pipeline model: {e}")
            
            try:
                # Try to load the best model
                self.model = joblib.load(self.config.YIELD_MODEL)
                self.model_loaded = True
                print("[OK] Best model loaded successfully")
            except Exception as e2:
                print(f"[WARN] Error loading best model: {e2}")
                self.model_loaded = False
    
    def predict(self, input_data):
        """Predict crop yield based on input parameters"""
        if not self.model_loaded:
            return self._predict_fallback(input_data)
        
        try:
            # Prepare input DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Use pipeline if available
            if self.pipeline is not None:
                prediction = self.pipeline.predict(input_df)[0]
                confidence = 0.85  # Placeholder for confidence score
            elif self.model is not None:
                # Preprocess input for the model
                processed_input = self._preprocess_input(input_df)
                prediction = self.model.predict(processed_input)[0]
                confidence = 0.80
            else:
                return self._predict_fallback(input_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(input_data, prediction)
            
            return prediction, confidence, recommendations
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return self._predict_fallback(input_data)
    
    def _preprocess_input(self, input_df):
        """Preprocess input data for model prediction"""
        # This should match the preprocessing used during training
        numerical_cols = ['soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm',
                         'humidity_%', 'sunlight_hours', 'pesticide_usage_ml',
                         'NDVI_index', 'total_days']
        
        categorical_cols = ['region', 'crop_type', 'irrigation_type',
                           'fertilizer_type', 'crop_disease_status']
        
        # One-hot encode categorical variables
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols)
        
        # Ensure all expected columns are present
        expected_cols = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else []
        
        if len(expected_cols) > 0:
            # Add missing columns with 0
            for col in expected_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Reorder columns to match training
            input_encoded = input_encoded[expected_cols]
        
        return input_encoded
    
    def _predict_fallback(self, input_data):
        """Fallback prediction using rule-based system"""
        # Base yield for different crops
        base_yields = {
            'Wheat': 3500,
            'Corn': 4500,
            'Rice': 4200,
            'Soybean': 2800,
            'Barley': 3200
        }
        
        crop_type = input_data.get('crop_type', 'Wheat')
        base_yield = base_yields.get(crop_type, 3500)
        
        # Adjust based on conditions
        adjustments = 0
        
        # Soil moisture adjustment
        soil_moisture = input_data.get('soil_moisture_%', 50)
        if soil_moisture < 30:
            adjustments -= 0.3
        elif soil_moisture > 70:
            adjustments -= 0.2
        elif 40 <= soil_moisture <= 60:
            adjustments += 0.1
        
        # pH adjustment
        soil_pH = input_data.get('soil_pH', 6.5)
        if 6.0 <= soil_pH <= 7.0:
            adjustments += 0.1
        elif soil_pH < 5.5 or soil_pH > 7.5:
            adjustments -= 0.2
        
        # Disease adjustment
        disease = input_data.get('crop_disease_status', 'No Disease')
        if disease == 'Severe':
            adjustments -= 0.4
        elif disease == 'Moderate':
            adjustments -= 0.2
        elif disease == 'Mild':
            adjustments -= 0.1
        
        # Irrigation adjustment
        irrigation = input_data.get('irrigation_type', 'No Irrigation')
        if irrigation == 'Drip':
            adjustments += 0.15
        elif irrigation == 'Sprinkler':
            adjustments += 0.1
        
        # Calculate final prediction
        prediction = base_yield * (1 + adjustments)
        confidence = 0.65
        
        # Generate recommendations
        recommendations = self._generate_recommendations(input_data, prediction)
        
        return prediction, confidence, recommendations
    
    def _generate_recommendations(self, input_data, predicted_yield):
        """Generate recommendations for improving yield"""
        recommendations = []
        
        # Check soil moisture
        soil_moisture = input_data.get('soil_moisture_%', 50)
        if soil_moisture < 30:
            recommendations.append({
                'category': 'Irrigation',
                'message': 'Soil moisture is very low. Increase irrigation.',
                'priority': 'High',
                'expected_impact': '+15-20%'
            })
        elif soil_moisture > 70:
            recommendations.append({
                'category': 'Drainage',
                'message': 'Soil moisture is very high. Improve drainage.',
                'priority': 'Medium',
                'expected_impact': '+5-10%'
            })
        
        # Check soil pH
        soil_pH = input_data.get('soil_pH', 6.5)
        if soil_pH < 6.0:
            recommendations.append({
                'category': 'Soil Management',
                'message': 'Soil pH is acidic. Consider adding lime.',
                'priority': 'High',
                'expected_impact': '+10-15%'
            })
        elif soil_pH > 7.5:
            recommendations.append({
                'category': 'Soil Management',
                'message': 'Soil pH is alkaline. Consider adding sulfur.',
                'priority': 'High',
                'expected_impact': '+10-15%'
            })
        
        # Check disease status
        disease = input_data.get('crop_disease_status', 'No Disease')
        if disease != 'No Disease':
            recommendations.append({
                'category': 'Disease Control',
                'message': f'Crop disease detected ({disease}). Apply appropriate treatment.',
                'priority': 'High',
                'expected_impact': '+20-30%'
            })
        
        # Check NDVI
        ndvi = input_data.get('NDVI_index', 0.5)
        if ndvi < 0.4:
            recommendations.append({
                'category': 'Crop Health',
                'message': 'Low vegetation health (NDVI). Check nutrient levels.',
                'priority': 'Medium',
                'expected_impact': '+10-15%'
            })
        
        # Check irrigation type
        irrigation = input_data.get('irrigation_type', 'No Irrigation')
        if irrigation == 'No Irrigation' or irrigation == 'Flood':
            recommendations.append({
                'category': 'Irrigation',
                'message': 'Consider upgrading to drip or sprinkler irrigation for better efficiency.',
                'priority': 'Medium',
                'expected_impact': '+10-20%'
            })
        
        # Add general recommendations if less than 3 specific ones
        if len(recommendations) < 3:
            general_recommendations = [
                {
                    'category': 'Fertilization',
                    'message': 'Regular soil testing and balanced fertilization can improve yield.',
                    'priority': 'Medium',
                    'expected_impact': '+10-15%'
                },
                {
                    'category': 'Crop Rotation',
                    'message': 'Implement crop rotation to maintain soil health.',
                    'priority': 'Low',
                    'expected_impact': '+5-10%'
                },
                {
                    'category': 'Pest Management',
                    'message': 'Regular monitoring and integrated pest management.',
                    'priority': 'Medium',
                    'expected_impact': '+5-15%'
                }
            ]
            recommendations.extend(general_recommendations[:3-len(recommendations)])
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def find_similar_farms(self, input_data, df, n_neighbors=5):
        """Find similar farms in the dataset"""
        try:
            # Select numerical features for similarity
            numerical_features = ['soil_moisture_%', 'soil_pH', 'temperature_C', 
                                 'rainfall_mm', 'humidity_%', 'sunlight_hours',
                                 'NDVI_index', 'total_days']
            
            # Filter features that exist in both input and dataframe
            available_features = [f for f in numerical_features 
                                 if f in input_data and f in df.columns]
            
            if len(available_features) < 3:
                return []
            
            # Extract input values
            input_values = [[input_data[f] for f in available_features]]
            
            # Extract dataset values
            dataset_values = df[available_features].values
            
            # Scale the data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            dataset_scaled = scaler.fit_transform(dataset_values)
            input_scaled = scaler.transform(input_values)
            
            # Find nearest neighbors
            nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(dataset_scaled)))
            nn.fit(dataset_scaled)
            distances, indices = nn.kneighbors(input_scaled)
            
            # Get similar farms data
            similar_farms = []
            for i, idx in enumerate(indices[0]):
                farm_data = df.iloc[idx].to_dict()
                similar_farms.append({
                    'id': int(farm_data.get('farm_id', idx)),
                    'region': farm_data.get('region', 'Unknown'),
                    'crop_type': farm_data.get('crop_type', 'Unknown'),
                    'yield': round(farm_data.get('yield_kg_per_hectare', 0), 2),
                    'similarity_score': round(1 - distances[0][i] / distances[0].max(), 3),
                    'soil_moisture': round(farm_data.get('soil_moisture_%', 0), 1),
                    'soil_pH': round(farm_data.get('soil_pH', 0), 1)
                })
            
            return similar_farms
            
        except Exception as e:
            print(f"Error finding similar farms: {e}")
            return []