from flask import Flask, render_template, jsonify, request, send_file, session
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from config import Config
from utils.data_loader import DataLoader
from utils.model_predictor import ModelPredictor
from utils.visualization import Visualization

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize utilities
data_loader = DataLoader()
model_predictor = ModelPredictor()
viz = Visualization()

# Global variables
df_smart_farming = None
df_commodity = None
df_fruit = None
df_vegetable = None

def load_data():
    """Load all datasets"""
    global df_smart_farming, df_commodity, df_fruit, df_vegetable
    
    try:
        df_smart_farming = data_loader.load_smart_farming_data()
        df_commodity = data_loader.load_commodity_data()
        df_fruit = data_loader.load_fruit_data()
        df_vegetable = data_loader.load_vegetable_data()
        print("All data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {e}")

# Load data on startup
with app.app_context():
    load_data()

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page"""
    if df_smart_farming is None:
        load_data()
    
    # Basic statistics for home page
    stats = {
        'total_farms': len(df_smart_farming),
        'total_crops': df_smart_farming['crop_type'].nunique(),
        'total_regions': df_smart_farming['region'].nunique(),
        'avg_yield': round(df_smart_farming['yield_kg_per_hectare'].mean(), 2),
        'max_yield': round(df_smart_farming['yield_kg_per_hectare'].max(), 2),
        'min_yield': round(df_smart_farming['yield_kg_per_hectare'].min(), 2)
    }
    
    # Top performing crops
    top_crops = df_smart_farming.groupby('crop_type')['yield_kg_per_hectare'].mean().nlargest(5).round(2)
    top_crops = top_crops.to_dict()
    
    return render_template('index.html', stats=stats, top_crops=top_crops)

@app.route('/dashboard')
def dashboard():
    """Interactive dashboard"""
    return render_template('dashboard.html')

@app.route('/predictions')
def predictions():
    """Yield predictions page"""
    # Get unique values for dropdowns
    regions = sorted(df_smart_farming['region'].unique().tolist()) if df_smart_farming is not None else []
    crop_types = sorted(df_smart_farming['crop_type'].unique().tolist()) if df_smart_farming is not None else []
    irrigation_types = sorted(df_smart_farming['irrigation_type'].unique().tolist()) if df_smart_farming is not None else []
    fertilizer_types = sorted(df_smart_farming['fertilizer_type'].unique().tolist()) if df_smart_farming is not None else []
    
    return render_template('predictions.html', 
                         regions=regions, 
                         crop_types=crop_types,
                         irrigation_types=irrigation_types,
                         fertilizer_types=fertilizer_types)

@app.route('/analysis')
def analysis():
    """Data analysis page"""
    return render_template('analysis.html')

@app.route('/data-explorer')
def data_explorer():
    """Interactive data explorer page"""
    return render_template('data_explorer.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

# ==================== API ENDPOINTS ====================

@app.route('/api/data/stats')
def get_data_stats():
    """Get comprehensive statistics"""
    try:
        if df_smart_farming is None:
            load_data()
        
        stats = {
            'success': True,
            'dataset_info': {
                'total_records': len(df_smart_farming),
                'total_columns': len(df_smart_farming.columns),
                'columns': df_smart_farming.columns.tolist()
            },
            'yield_statistics': {
                'mean': float(df_smart_farming['yield_kg_per_hectare'].mean()),
                'median': float(df_smart_farming['yield_kg_per_hectare'].median()),
                'std': float(df_smart_farming['yield_kg_per_hectare'].std()),
                'min': float(df_smart_farming['yield_kg_per_hectare'].min()),
                'max': float(df_smart_farming['yield_kg_per_hectare'].max())
            },
            'categorical_counts': {
                'region': {
                    'unique_values': int(df_smart_farming['region'].nunique()),
                    'values': df_smart_farming['region'].value_counts().head(10).to_dict()
                },
                'crop_type': {
                    'unique_values': int(df_smart_farming['crop_type'].nunique()),
                    'values': df_smart_farming['crop_type'].value_counts().head(10).to_dict()
                }
            }
        }
        
        return jsonify(stats)
    except Exception as e:
        print(f"Error in get_data_stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/api/data/overview')
def get_data_overview():
    """Get data overview for dashboard"""
    if df_smart_farming is None:
        load_data()
    
    overview = {
        'total_records': len(df_smart_farming),
        'columns': df_smart_farming.columns.tolist(),
        'data_types': {col: str(dtype) for col, dtype in df_smart_farming.dtypes.items()},
        'missing_values': df_smart_farming.isnull().sum().to_dict(),
        'sample_data': df_smart_farming.head(10).to_dict('records')
    }
    
    return jsonify(overview)

@app.route('/api/charts/yield-distribution')
def get_yield_distribution():
    """Get yield distribution chart"""
    try:
        if df_smart_farming is None:
            load_data()
        
        # Create histogram data
        yield_data = df_smart_farming['yield_kg_per_hectare'].dropna()
        
        trace = {
            'x': yield_data.tolist(),
            'type': 'histogram',
            'nbinsx': 30,
            'name': 'Yield Distribution',
            'marker': {
                'color': '#2ecc71',
                'opacity': 0.7
            }
        }
        
        layout = {
            'title': 'Distribution of Crop Yield',
            'xaxis': {'title': 'Yield (kg/ha)'},
            'yaxis': {'title': 'Frequency'},
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#333'}
        }
        
        return jsonify({'success': True, 'data': [trace], 'layout': layout})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/charts/crop-performance')
def get_crop_performance():
    """Get crop performance comparison"""
    try:
        if df_smart_farming is None:
            load_data()
        
        # Calculate average yield by crop
        crop_stats = df_smart_farming.groupby('crop_type')['yield_kg_per_hectare'].mean().reset_index()
        crop_stats = crop_stats.sort_values('yield_kg_per_hectare', ascending=True)
        
        trace = {
            'x': crop_stats['yield_kg_per_hectare'].tolist(),
            'y': crop_stats['crop_type'].tolist(),
            'type': 'bar',
            'orientation': 'h',
            'name': 'Average Yield',
            'marker': {
                'color': '#3498db',
                'opacity': 0.7
            }
        }
        
        layout = {
            'title': 'Average Yield by Crop Type',
            'xaxis': {'title': 'Average Yield (kg/ha)'},
            'yaxis': {'title': 'Crop Type'},
            'height': 400,
            'margin': {'l': 150},
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#333'}
        }
        
        return jsonify({'success': True, 'data': [trace], 'layout': layout})
    except Exception as e:
        print(f"Error in get_crop_performance: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/charts/region-analysis')
def get_region_analysis():
    """Get regional analysis"""
    try:
        if df_smart_farming is None:
            load_data()
        
        # Calculate statistics by region
        region_stats = df_smart_farming.groupby('region').agg({
            'yield_kg_per_hectare': 'mean',
            'temperature_C': 'mean'
        }).reset_index()
        
        # Create traces
        trace1 = {
            'x': region_stats['region'].tolist(),
            'y': region_stats['yield_kg_per_hectare'].tolist(),
            'type': 'bar',
            'name': 'Avg Yield',
            'marker': {'color': '#2ecc71'}
        }
        
        trace2 = {
            'x': region_stats['region'].tolist(),
            'y': region_stats['temperature_C'].tolist(),
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Avg Temperature',
            'yaxis': 'y2',
            'line': {'color': '#e74c3c'}
        }
        
        layout = {
            'title': 'Regional Analysis',
            'xaxis': {'title': 'Region'},
            'yaxis': {
                'title': 'Avg Yield (kg/ha)',
                'titlefont': {'color': '#2ecc71'},
                'tickfont': {'color': '#2ecc71'}
            },
            'yaxis2': {
                'title': 'Avg Temperature (°C)',
                'titlefont': {'color': '#e74c3c'},
                'tickfont': {'color': '#e74c3c'},
                'overlaying': 'y',
                'side': 'right'
            },
            'height': 400,
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#333'}
        }
        
        return jsonify({'success': True, 'data': [trace1, trace2], 'layout': layout})
    except Exception as e:
        print(f"Error in get_region_analysis: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/charts/correlation-matrix')
def get_correlation_matrix():
    """Get correlation matrix heatmap"""
    if df_smart_farming is None:
        load_data()
    
    # Select numerical columns
    numerical_cols = df_smart_farming.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df_smart_farming[numerical_cols].corr().round(2)
    
    fig = px.imshow(corr_matrix,
                   text_auto=True,
                   aspect="auto",
                   color_continuous_scale='RdBu_r',
                   title='Correlation Matrix',
                   labels=dict(color="Correlation"))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333')
    )
    
    return jsonify(fig.to_dict())

@app.route('/api/charts/time-trends')
def get_time_trends():
    """Get time-based trends"""
    if df_smart_farming is None:
        load_data()
    
    # Check if date columns exist
    if 'sowing_date' in df_smart_farming.columns and 'harvest_date' in df_smart_farming.columns:
        df_smart_farming['sowing_date'] = pd.to_datetime(df_smart_farming['sowing_date'])
        df_smart_farming['sowing_month'] = df_smart_farming['sowing_date'].dt.month
        
        monthly_yield = df_smart_farming.groupby('sowing_month')['yield_kg_per_hectare'].mean().reset_index()
        
        fig = px.line(monthly_yield, 
                     x='sowing_month', 
                     y='yield_kg_per_hectare',
                     markers=True,
                     title='Average Yield by Sowing Month',
                     labels={'sowing_month': 'Month', 'yield_kg_per_hectare': 'Average Yield (kg/ha)'})
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333')
        )
        
        return jsonify(fig.to_dict())
    
    return jsonify({'error': 'Date columns not found'})

@app.route('/api/predict', methods=['POST'])
def predict_yield():
    """Predict crop yield based on input parameters"""
    try:
        data = request.json
        
        # Log received data
        print("Received prediction request:", data)
        
        # Validate required fields with defaults
        required_fields = {
            'soil_moisture': 50.0,
            'soil_pH': 6.5,
            'temperature': 25.0,
            'rainfall': 500.0,
            'humidity': 65.0,
            'sunlight_hours': 8.0,
            'pesticide_usage': 100.0,
            'NDVI_index': 0.7,
            'total_days': 120,
            'region': 'North',
            'crop_type': 'Wheat',
            'irrigation_type': 'Drip',
            'fertilizer_type': 'Organic',
            'disease_status': 'No Disease'
        }
        
        # Fill missing fields with defaults
        for field, default_value in required_fields.items():
            if field not in data or data[field] is None:
                data[field] = default_value
        
        # Try to use the ML model
        try:
            if model_predictor.model_loaded:
                # Prepare input for model
                input_data = {
                    'soil_moisture_%': float(data['soil_moisture']),
                    'soil_pH': float(data['soil_pH']),
                    'temperature_C': float(data['temperature']),
                    'rainfall_mm': float(data['rainfall']),
                    'humidity_%': float(data['humidity']),
                    'sunlight_hours': float(data['sunlight_hours']),
                    'pesticide_usage_ml': float(data['pesticide_usage']),
                    'NDVI_index': float(data['NDVI_index']),
                    'total_days': int(data['total_days']),
                    'region': data['region'],
                    'crop_type': data['crop_type'],
                    'irrigation_type': data['irrigation_type'],
                    'fertilizer_type': data['fertilizer_type'],
                    'crop_disease_status': data['disease_status']
                }
                
                prediction, confidence, recommendations = model_predictor.predict(input_data)
                
                # Find similar farms
                similar_farms = model_predictor.find_similar_farms(input_data, df_smart_farming)
                
                # If no similar farms found, use fallback
                if not similar_farms or len(similar_farms) == 0:
                    similar_farms = get_fallback_similar_farms(input_data, df_smart_farming)
                
                response = {
                    'success': True,
                    'prediction': float(prediction),
                    'confidence': float(confidence),
                    'recommendations': recommendations,
                    'similar_farms': similar_farms[:5],  # Limit to 5 farms
                    'model_used': 'ML Model',
                    'input_data': input_data  # Add input data for reference
                }
                
                print(f"ML Prediction: {prediction} kg/ha, Similar farms found: {len(similar_farms)}")
                return jsonify(response)
            
        except Exception as ml_error:
            print(f"ML model error, using fallback: {ml_error}")
        
        # Fallback: Rule-based prediction
        prediction, confidence, recommendations = model_predictor._predict_fallback({
            'soil_moisture_%': float(data['soil_moisture']),
            'soil_pH': float(data['soil_pH']),
            'crop_type': data['crop_type'],
            'crop_disease_status': data['disease_status'],
            'irrigation_type': data['irrigation_type']
        })
        
        # Prepare input data for similarity search
        input_data = {
            'soil_moisture_%': float(data['soil_moisture']),
            'soil_pH': float(data['soil_pH']),
            'temperature_C': float(data['temperature']),
            'rainfall_mm': float(data['rainfall']),
            'NDVI_index': float(data['NDVI_index']),
            'region': data['region'],
            'crop_type': data['crop_type'],
            'irrigation_type': data['irrigation_type'],
            'fertilizer_type': data['fertilizer_type'],
            'crop_disease_status': data['disease_status']
        }
        
        # Find similar farms using fallback method
        similar_farms = get_fallback_similar_farms(input_data, df_smart_farming)
        
        response = {
            'success': True,
            'prediction': float(prediction),
            'confidence': float(confidence),
            'recommendations': recommendations,
            'similar_farms': similar_farms[:3],
            'model_used': 'Rule-Based Fallback',
            'input_data': input_data
        }
        
        print(f"Fallback Prediction: {prediction} kg/ha, Similar farms found: {len(similar_farms)}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        
        # Ultimate fallback: return average yield
        avg_yield = df_smart_farming['yield_kg_per_hectare'].mean() if df_smart_farming is not None else 4000
        
        return jsonify({
            'success': True,
            'prediction': float(avg_yield),
            'confidence': 0.5,
            'recommendations': [{
                'category': 'System',
                'message': 'Using average yield estimate. ML model not available.',
                'priority': 'Low',
                'expected_impact': 'N/A'
            }],
            'similar_farms': [],
            'model_used': 'Average Yield'
        })


def get_fallback_similar_farms(input_data, df):
    """Fallback method to find similar farms when ML model doesn't find any"""
    try:
        # Create a copy of the dataframe
        similar = df.copy()
        
        # Start with a base similarity score
        similar['similarity_score'] = 0.5
        
        # Apply filters and adjust similarity scores
        if 'crop_type' in input_data and input_data['crop_type']:
            # Exact match for crop type gives higher score
            similar.loc[similar['crop_type'] == input_data['crop_type'], 'similarity_score'] += 0.3
            similar = similar[similar['crop_type'] == input_data['crop_type']]
        
        if 'region' in input_data and input_data['region']:
            # Exact match for region gives higher score
            similar.loc[similar['region'] == input_data['region'], 'similarity_score'] += 0.2
            similar = similar[similar['region'] == input_data['region']]
        
        # Adjust for numerical similarities
        numerical_features = ['soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm', 'NDVI_index']
        
        for feature in numerical_features:
            if feature in input_data and feature in similar.columns:
                try:
                    input_value = float(input_data[feature])
                    # Calculate similarity based on percentage difference
                    similar['feature_diff'] = abs(similar[feature] - input_value) / max(input_value, 1)
                    # Adjust similarity score (smaller difference = higher score)
                    similar['similarity_score'] -= similar['feature_diff'] * 0.1
                except:
                    pass
        
        # Ensure similarity score is between 0.1 and 0.95
        similar['similarity_score'] = similar['similarity_score'].clip(lower=0.1, upper=0.95)
        
        # If we have no results after filtering, get some random farms with default scores
        if len(similar) == 0:
            similar = df.sample(min(10, len(df))).copy()
            similar['similarity_score'] = 0.5
        
        # Sort by similarity score (highest first)
        similar = similar.sort_values('similarity_score', ascending=False)
        
        # Convert to list of dictionaries
        result = []
        for _, farm in similar.head(10).iterrows():  # Get top 10 most similar
            farm_dict = farm.to_dict()
            
            # Ensure all fields are JSON serializable
            for key, value in farm_dict.items():
                if pd.isna(value):
                    farm_dict[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    farm_dict[key] = float(value)
                elif isinstance(value, np.ndarray):
                    farm_dict[key] = value.tolist()
            
            result.append(farm_dict)
        
        print(f"Fallback found {len(result)} similar farms")
        return result
        
    except Exception as e:
        print(f"Error in get_fallback_similar_farms: {e}")
        import traceback
        traceback.print_exc()
        
        # Return some sample farms as a last resort
        try:
            sample_farms = df.sample(min(5, len(df))).to_dict('records')
            for farm in sample_farms:
                farm['similarity_score'] = 0.5
            return sample_farms
        except:
            return []

@app.route('/api/data/crop-stats/<crop_type>', methods=['GET'])
def get_crop_stats(crop_type):
    """Get statistics for a specific crop type"""
    try:
        if df_smart_farming is None:
            load_data()
        
        # Filter by crop type
        crop_data = df_smart_farming[df_smart_farming['crop_type'] == crop_type]
        
        if len(crop_data) == 0:
            return jsonify({
                'success': False,
                'error': f'No data found for crop type: {crop_type}'
            }), 404
        
        stats = {
            'success': True,
            'crop_type': crop_type,
            'count': len(crop_data),
            'yield_stats': {
                'mean': float(crop_data['yield_kg_per_hectare'].mean()),
                'median': float(crop_data['yield_kg_per_hectare'].median()),
                'std': float(crop_data['yield_kg_per_hectare'].std()),
                'min': float(crop_data['yield_kg_per_hectare'].min()),
                'max': float(crop_data['yield_kg_per_hectare'].max()),
                'q1': float(crop_data['yield_kg_per_hectare'].quantile(0.25)),
                'q3': float(crop_data['yield_kg_per_hectare'].quantile(0.75))
            },
            'region_distribution': crop_data['region'].value_counts().to_dict(),
            'irrigation_distribution': crop_data['irrigation_type'].value_counts().to_dict()
        }
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"Error in get_crop_stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data/region-stats/<region>', methods=['GET'])
def get_region_stats(region):
    """Get statistics for a specific region"""
    try:
        if df_smart_farming is None:
            load_data()
        
        # Filter by region
        region_data = df_smart_farming[df_smart_farming['region'] == region]
        
        if len(region_data) == 0:
            return jsonify({
                'success': False,
                'error': f'No data found for region: {region}'
            }), 404
        
        stats = {
            'success': True,
            'region': region,
            'count': len(region_data),
            'yield_stats': {
                'mean': float(region_data['yield_kg_per_hectare'].mean()),
                'median': float(region_data['yield_kg_per_hectare'].median()),
                'std': float(region_data['yield_kg_per_hectare'].std()),
                'min': float(region_data['yield_kg_per_hectare'].min()),
                'max': float(region_data['yield_kg_per_hectare'].max())
            },
            'crop_distribution': region_data['crop_type'].value_counts().to_dict(),
            'top_performing_crops': region_data.groupby('crop_type')['yield_kg_per_hectare']
                                              .mean()
                                              .nlargest(5)
                                              .to_dict()
        }
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"Error in get_region_stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/api/data/filters')
def get_filter_options():
    """Get filter options for dropdowns"""
    try:
        if df_smart_farming is None:
            load_data()
        
        # Get unique values from dataframe
        filters = {
            'regions': sorted([str(x) for x in df_smart_farming['region'].dropna().unique()]),
            'crop_types': sorted([str(x) for x in df_smart_farming['crop_type'].dropna().unique()]),
            'irrigation_types': sorted([str(x) for x in df_smart_farming['irrigation_type'].dropna().unique()]),
            'disease_statuses': sorted([str(x) for x in df_smart_farming['crop_disease_status'].dropna().unique()])
        }
        
        return jsonify({'success': True, 'filters': filters})
    except Exception as e:
        print(f"Error in get_filter_options: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
        
@app.route('/api/data/filter', methods=['POST'])
def filter_data():
    """Filter data based on criteria"""
    try:
        # Check if request has data
        if not request.data:
            print("Warning: Empty request received")
            # Return all data if no filters provided
            result = {
                'success': True,
                'total_records': len(df_smart_farming),
                'average_yield': float(df_smart_farming['yield_kg_per_hectare'].mean()),
                'data': df_smart_farming.head(100).to_dict('records')
            }
            return jsonify(result)
        
        # Try to parse JSON
        try:
            filters = request.json or {}
        except Exception as json_error:
            print(f"JSON parsing error: {json_error}")
            # If JSON is invalid, treat as empty filters
            filters = {}
        
        print(f"Received filters: {filters}")
        
        if df_smart_farming is None:
            load_data()
        
        filtered_df = df_smart_farming.copy()
        
        # Apply filters
        if filters.get('region') and filters['region'] != '':
            filtered_df = filtered_df[filtered_df['region'] == filters['region']]
        
        if filters.get('crop_type') and filters['crop_type'] != '':
            filtered_df = filtered_df[filtered_df['crop_type'] == filters['crop_type']]
        
        if filters.get('irrigation_type') and filters['irrigation_type'] != '':
            filtered_df = filtered_df[filtered_df['irrigation_type'] == filters['irrigation_type']]
        
        if filters.get('min_yield') and filters['min_yield'] != '':
            try:
                min_yield = float(filters['min_yield'])
                filtered_df = filtered_df[filtered_df['yield_kg_per_hectare'] >= min_yield]
            except ValueError:
                pass  # Ignore invalid numbers
        
        if filters.get('max_yield') and filters['max_yield'] != '':
            try:
                max_yield = float(filters['max_yield'])
                filtered_df = filtered_df[filtered_df['yield_kg_per_hectare'] <= max_yield]
            except ValueError:
                pass  # Ignore invalid numbers
        
        # Convert to JSON serializable format
        result = {
            'success': True,
            'total_records': len(filtered_df),
            'average_yield': float(filtered_df['yield_kg_per_hectare'].mean()) if len(filtered_df) > 0 else 0,
            'data': filtered_df.head(100).to_dict('records')  # Limit to 100 records
        }
        
        print(f"Filtered data: {len(filtered_df)} records")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in filter_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    
    
@app.route('/api/data/analysis', methods=['GET'])
def get_analysis_data():
    """Get data specifically formatted for analysis"""
    try:
        if df_smart_farming is None:
            load_data()
        
        # Get a sample of data for analysis (limit to 5000 records for performance)
        sample_size = min(5000, len(df_smart_farming))
        analysis_df = df_smart_farming.sample(sample_size, random_state=42) if sample_size < len(df_smart_farming) else df_smart_farming
        
        # Convert to JSON serializable format
        result = {
            'success': True,
            'total_records': len(analysis_df),
            'data': analysis_df.to_dict('records')
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in get_analysis_data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    

@app.route('/api/data/all', methods=['GET'])
def get_all_data():
    """Get all data for initial load"""
    try:
        if df_smart_farming is None:
            load_data()
        
        result = {
            'success': True,
            'total_records': len(df_smart_farming),
            'average_yield': float(df_smart_farming['yield_kg_per_hectare'].mean()),
            'data': df_smart_farming.to_dict('records')  # Return ALL records
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
     
@app.route('/api/export/csv')
def export_csv():
    """Export filtered data as CSV"""
    try:
        # Get filter parameters from query string
        region = request.args.get('region', '')
        crop_type = request.args.get('crop_type', '')
        
        filtered_df = df_smart_farming.copy()
        
        if region:
            filtered_df = filtered_df[filtered_df['region'] == region]
        if crop_type:
            filtered_df = filtered_df[filtered_df['crop_type'] == crop_type]
        
        # Create CSV in memory
        csv_data = filtered_df.to_csv(index=False)
        
        # Create response
        output = io.StringIO()
        output.write(csv_data)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'smart_farming_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualization/scatter')
def get_scatter_plot():
    """Generate scatter plot visualization"""
    try:
        x_var = request.args.get('x', 'soil_moisture_%')
        y_var = request.args.get('y', 'yield_kg_per_hectare')
        color_var = request.args.get('color', 'crop_type')
        
        fig = px.scatter(df_smart_farming, 
                        x=x_var, 
                        y=y_var,
                        color=color_var,
                        hover_data=['region', 'crop_type'],
                        title=f'{y_var} vs {x_var} by {color_var}',
                        trendline='ols' if x_var != color_var else None)
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333')
        )
        
        return jsonify(fig.to_dict())
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/debug', methods=['GET'])
def debug_data():
    """Debug endpoint to see what data exists"""
    try:
        if df_smart_farming is None:
            load_data()
        
        # Get sample of each category
        debug_info = {
            'regions': df_smart_farming['region'].unique().tolist(),
            'crop_types': df_smart_farming['crop_type'].unique().tolist(),
            'irrigation_types': df_smart_farming['irrigation_type'].unique().tolist(),
            'yield_range': {
                'min': float(df_smart_farming['yield_kg_per_hectare'].min()),
                'max': float(df_smart_farming['yield_kg_per_hectare'].max()),
                'mean': float(df_smart_farming['yield_kg_per_hectare'].mean()),
                'median': float(df_smart_farming['yield_kg_per_hectare'].median())
            },
            'sample_combinations': []
        }
        
        # Sample some actual combinations from data
        sample_df = df_smart_farming.head(10)
        for _, row in sample_df.iterrows():
            debug_info['sample_combinations'].append({
                'region': row['region'],
                'crop_type': row['crop_type'],
                'irrigation_type': row['irrigation_type'],
                'yield': row['yield_kg_per_hectare']
            })
        
        return jsonify({'success': True, 'debug': debug_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    print("Starting Smart Farming Dashboard...")
    print(f"Data loaded: {len(df_smart_farming) if df_smart_farming is not None else 0} records")
    print(f"Model loaded: {model_predictor.model_loaded}")
    app.run(debug=True, host='0.0.0.0', port=5000)