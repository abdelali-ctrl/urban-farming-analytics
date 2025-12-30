import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import base64
from io import BytesIO

class Visualization:
    def __init__(self):
        self.color_palette = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', 
                             '#f39c12', '#1abc9c', '#34495e', '#e67e22']
        
    def create_yield_distribution(self, df):
        """Create yield distribution histogram"""
        fig = px.histogram(df, 
                          x='yield_kg_per_hectare',
                          nbins=30,
                          title='Distribution of Crop Yield',
                          labels={'yield_kg_per_hectare': 'Yield (kg/ha)'},
                          color_discrete_sequence=[self.color_palette[0]])
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333'),
            bargap=0.1
        )
        
        return fig
    
    def create_crop_comparison(self, df):
        """Create crop comparison chart"""
        crop_stats = df.groupby('crop_type')['yield_kg_per_hectare'].agg(['mean', 'std', 'count']).reset_index()
        crop_stats = crop_stats.round(2)
        
        fig = px.bar(crop_stats, 
                    x='crop_type', 
                    y='mean',
                    error_y='std',
                    title='Average Yield by Crop Type',
                    labels={'mean': 'Average Yield (kg/ha)', 'crop_type': 'Crop Type'},
                    color='mean',
                    color_continuous_scale='Viridis',
                    hover_data=['count'])
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333'),
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_region_analysis(self, df):
        """Create regional analysis dashboard"""
        region_stats = df.groupby('region').agg({
            'yield_kg_per_hectare': ['mean', 'count'],
            'temperature_C': 'mean',
            'rainfall_mm': 'mean',
            'NDVI_index': 'mean'
        }).round(2)
        
        region_stats.columns = ['avg_yield', 'farm_count', 'avg_temp', 'avg_rainfall', 'avg_ndvi']
        region_stats = region_stats.reset_index()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Average Yield', 'Number of Farms', 'Average Temperature',
                           'Average Rainfall', 'Average NDVI', 'Yield vs Temperature'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Bar charts
        fig.add_trace(
            go.Bar(x=region_stats['region'], y=region_stats['avg_yield'], 
                   name='Avg Yield', marker_color=self.color_palette[0]),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=region_stats['region'], y=region_stats['farm_count'], 
                   name='Farm Count', marker_color=self.color_palette[1]),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=region_stats['region'], y=region_stats['avg_temp'], 
                   name='Avg Temp', marker_color=self.color_palette[2]),
            row=1, col=3
        )
        
        fig.add_trace(
            go.Bar(x=region_stats['region'], y=region_stats['avg_rainfall'], 
                   name='Avg Rainfall', marker_color=self.color_palette[3]),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=region_stats['region'], y=region_stats['avg_ndvi'], 
                   name='Avg NDVI', marker_color=self.color_palette[4]),
            row=2, col=2
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(x=region_stats['avg_temp'], y=region_stats['avg_yield'],
                      mode='markers+text',
                      text=region_stats['region'],
                      textposition="top center",
                      marker=dict(size=region_stats['farm_count']/10, 
                                 color=region_stats['avg_yield'],
                                 colorscale='Viridis',
                                 showscale=True),
                      name='Yield vs Temp'),
            row=2, col=3
        )
        
        fig.update_layout(height=800, showlegend=False,
                         plot_bgcolor='rgba(0,0,0,0)',
                         paper_bgcolor='rgba(0,0,0,0)')
        
        return fig
    
    def create_correlation_heatmap(self, df):
        """Create correlation matrix heatmap"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = df[numerical_cols].corr().round(2)
        
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
        
        return fig
    
    def create_scatter_matrix(self, df):
        """Create scatter matrix for numerical features"""
        numerical_cols = ['soil_moisture_%', 'temperature_C', 'rainfall_mm',
                         'NDVI_index', 'yield_kg_per_hectare']
        
        available_cols = [col for col in numerical_cols if col in df.columns]
        
        if len(available_cols) >= 3:
            fig = px.scatter_matrix(df[available_cols],
                                   dimensions=available_cols,
                                   color=df['crop_type'] if 'crop_type' in df.columns else None,
                                   title='Scatter Matrix of Key Features',
                                   height=800)
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#333')
            )
            
            return fig
        
        return None
    
    def create_time_series(self, df):
        """Create time series analysis if date columns exist"""
        if 'sowing_date' in df.columns:
            df_date = df.copy()
            df_date['sowing_date'] = pd.to_datetime(df_date['sowing_date'])
            df_date['sowing_month'] = df_date['sowing_date'].dt.month
            
            monthly_stats = df_date.groupby('sowing_month').agg({
                'yield_kg_per_hectare': 'mean',
                'temperature_C': 'mean',
                'rainfall_mm': 'mean'
            }).reset_index()
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Average Yield by Month', 'Climate Factors by Month'),
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            # Yield by month
            fig.add_trace(
                go.Scatter(x=monthly_stats['sowing_month'], 
                          y=monthly_stats['yield_kg_per_hectare'],
                          mode='lines+markers',
                          name='Yield',
                          line=dict(color=self.color_palette[0], width=3)),
                row=1, col=1
            )
            
            # Climate factors
            fig.add_trace(
                go.Scatter(x=monthly_stats['sowing_month'], 
                          y=monthly_stats['temperature_C'],
                          mode='lines+markers',
                          name='Temperature',
                          line=dict(color=self.color_palette[2], width=2)),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(x=monthly_stats['sowing_month'], 
                      y=monthly_stats['rainfall_mm'],
                      name='Rainfall',
                      marker_color=self.color_palette[1],
                      opacity=0.6),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#333'),
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Month", row=2, col=1)
            fig.update_yaxes(title_text="Yield (kg/ha)", row=1, col=1)
            fig.update_yaxes(title_text="Value", row=2, col=1)
            
            return fig
        
        return None
    
    def create_disease_impact(self, df):
        """Create visualization of disease impact on yield"""
        if 'crop_disease_status' in df.columns:
            disease_stats = df.groupby('crop_disease_status')['yield_kg_per_hectare'].agg(['mean', 'std', 'count']).reset_index()
            
            # Order disease status
            disease_order = ['No Disease', 'Mild', 'Moderate', 'Severe']
            disease_stats['crop_disease_status'] = pd.Categorical(
                disease_stats['crop_disease_status'], 
                categories=disease_order,
                ordered=True
            )
            disease_stats = disease_stats.sort_values('crop_disease_status')
            
            fig = px.bar(disease_stats, 
                        x='crop_disease_status', 
                        y='mean',
                        error_y='std',
                        title='Impact of Disease on Crop Yield',
                        labels={'mean': 'Average Yield (kg/ha)', 'crop_disease_status': 'Disease Status'},
                        color='mean',
                        color_continuous_scale='RdYlGn_r',
                        hover_data=['count'])
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#333')
            )
            
            return fig
        
        return None
    
    def plot_to_base64(self, fig):
        """Convert plotly figure to base64 string"""
        img_bytes = fig.to_image(format="png", width=800, height=600)
        return base64.b64encode(img_bytes).decode('utf-8')