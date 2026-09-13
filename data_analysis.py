import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

class DataAnalyzer:
    def __init__(self, data_path):
        try:
            # Load your dataset
            self.df = pd.read_csv(data_path)
            print(f"✅ Dataset loaded successfully: {len(self.df)} songs")
            print(f"📊 File: {data_path}")
            self.clean_data()
            self.audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        except FileNotFoundError:
            print(f"❌ ERROR: File '{data_path}' not found!")
            print("💡 Make sure the CSV file is in the same folder as your Python files")
            return
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return
    
    def clean_data(self):
        """Clean and prepare the data for analysis"""
        print("🧹 Cleaning data...")
        
        # Handle missing values
        initial_count = len(self.df)
        self.df = self.df.dropna()
        print(f"   Removed {initial_count - len(self.df)} rows with missing values")
        
        # Remove duplicates based on track_id
        if 'track_id' in self.df.columns:
            initial_count = len(self.df)
            self.df = self.df.drop_duplicates(subset=['track_id'])
            print(f"   Removed {initial_count - len(self.df)} duplicate tracks")
        else:
            initial_count = len(self.df)
            self.df = self.df.drop_duplicates()
            print(f"   Removed {initial_count - len(self.df)} duplicate rows")
        
        # Create hit threshold (popularity > 70)
        self.df['is_hit'] = (self.df['popularity'] > 70).astype(int)
        print(f"🎯 Hit songs: {self.df['is_hit'].sum()}")
        print(f"🎵 Non-hit songs: {len(self.df) - self.df['is_hit'].sum()}")
        
        # Show column names to understand your data
        print("📋 Columns available:", list(self.df.columns))
        
        # Show first few rows
        print("\n📄 First 3 rows of data:")
        print(self.df.head(3))

    def get_basic_info(self):
        """Return basic dataset info for Streamlit display"""
        info = {
            'total_songs': len(self.df),
            'hit_songs': self.df['is_hit'].sum(),
            'non_hit_songs': len(self.df) - self.df['is_hit'].sum(),
            'unique_genres': self.df['track_genre'].nunique() if 'track_genre' in self.df.columns else 0,
            'columns': list(self.df.columns),
            'average_popularity': self.df['popularity'].mean(),
            'hit_threshold': 70
        }
        return info

    # ==================== VISUALIZATION METHODS ====================

    def create_feature_distribution(self, feature, show_hit_comparison=True):
        """Create distribution plot for a specific feature"""
        if feature not in self.df.columns:
            return None
        
        if show_hit_comparison:
            # Create subplots: histogram and box plot
            fig = make_subplots(rows=1, cols=2, 
                              subplot_titles=(f'{feature.title()} Distribution', 
                                            f'{feature.title()} - Hit vs Non-Hit'))
            
            # Histogram
            hit_data = self.df[self.df['is_hit'] == 1][feature]
            non_hit_data = self.df[self.df['is_hit'] == 0][feature]
            
            fig.add_trace(go.Histogram(x=hit_data, name='Hit Songs', 
                                     opacity=0.7, marker_color='#1DB954'), 1, 1)
            fig.add_trace(go.Histogram(x=non_hit_data, name='Non-Hit Songs', 
                                     opacity=0.7, marker_color='#FF4B4B'), 1, 1)
            
            # Box plot
            fig.add_trace(go.Box(y=hit_data, name='Hit Songs', 
                               marker_color='#1DB954', boxmean=True), 1, 2)
            fig.add_trace(go.Box(y=non_hit_data, name='Non-Hit Songs', 
                               marker_color='#FF4B4B', boxmean=True), 1, 2)
            
            fig.update_layout(barmode='overlay', showlegend=True)
        else:
            # Simple histogram
            fig = px.histogram(self.df, x=feature, 
                             title=f'{feature.title()} Distribution',
                             color_discrete_sequence=['#1DB954'])
        
        return fig

    def create_correlation_heatmap(self, features=None):
        """Create interactive correlation heatmap"""
        if features is None:
            features = self.audio_features + ['popularity']
        
        # Filter features that exist in dataframe
        features = [f for f in features if f in self.df.columns]
        
        corr_matrix = self.df[features].corr()
        
        fig = px.imshow(corr_matrix, 
                       title="Audio Features Correlation Heatmap",
                       color_continuous_scale='RdYlGn',
                       aspect="auto",
                       zmin=-1, zmax=1)
        
        # Add correlation values as annotations
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                fig.add_annotation(x=i, y=j, 
                                 text=f"{corr_matrix.iloc[i, j]:.2f}",
                                 showarrow=False,
                                 font=dict(color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white"))
        
        return fig

    def create_feature_scatter(self, x_feature, y_feature, color_by='is_hit'):
        """Create scatter plot between two features"""
        if x_feature not in self.df.columns or y_feature not in self.df.columns:
            return None
        
        if color_by == 'is_hit':
            fig = px.scatter(self.df, x=x_feature, y=y_feature, 
                           color='is_hit',
                           title=f"{x_feature.title()} vs {y_feature.title()}",
                           color_discrete_sequence=['#FF4B4B', '#1DB954'],
                           labels={'is_hit': 'Hit Song'},
                           hover_data=['track_name', 'artists', 'popularity'] if 'track_name' in self.df.columns else None)
        elif color_by == 'track_genre' and 'track_genre' in self.df.columns:
            # Show top 5 genres only for clarity
            top_genres = self.df['track_genre'].value_counts().head(5).index
            filtered_df = self.df[self.df['track_genre'].isin(top_genres)]
            fig = px.scatter(filtered_df, x=x_feature, y=y_feature, 
                           color='track_genre',
                           title=f"{x_feature.title()} vs {y_feature.title()} by Genre")
        else:
            fig = px.scatter(self.df, x=x_feature, y=y_feature,
                           title=f"{x_feature.title()} vs {y_feature.title()}")
        
        return fig

    def create_hit_profile_analysis(self):
        """Analyze what features characterize hit songs"""
        # Calculate average features for hits vs non-hits
        hit_avgs = self.df[self.df['is_hit'] == 1][self.audio_features].mean()
        non_hit_avgs = self.df[self.df['is_hit'] == 0][self.audio_features].mean()
        
        # Create comparison bar chart
        comparison_df = pd.DataFrame({
            'Feature': self.audio_features,
            'Hit Songs': hit_avgs.values,
            'Non-Hit Songs': non_hit_avgs.values
        })
        
        fig = px.bar(comparison_df, x='Feature', y=['Hit Songs', 'Non-Hit Songs'],
                    title="Hit vs Non-Hit Song Profile",
                    barmode='group',
                    color_discrete_sequence=['#1DB954', '#FF4B4B'])
        
        return fig

    def create_popularity_analysis(self):
        """Analyze popularity distribution and characteristics"""
        fig = make_subplots(rows=2, cols=2,
                          subplot_titles=('Popularity Distribution', 'Popularity vs Danceability',
                                        'Popularity vs Energy', 'Top Genres by Average Popularity'))
        
        # Popularity distribution
        fig.add_trace(go.Histogram(x=self.df['popularity'], name='Popularity',
                                 marker_color='#1DB954'), 1, 1)
        
        # Popularity vs Danceability
        fig.add_trace(go.Scatter(x=self.df['danceability'], y=self.df['popularity'],
                               mode='markers', name='Songs',
                               marker=dict(color=self.df['popularity'], 
                                         colorscale='Viridis', showscale=True),
                               hovertemplate='Danceability: %{x}<br>Popularity: %{y}'), 1, 2)
        
        # Popularity vs Energy
        fig.add_trace(go.Scatter(x=self.df['energy'], y=self.df['popularity'],
                               mode='markers', name='Songs',
                               marker=dict(color=self.df['popularity'], 
                                         colorscale='Viridis', showscale=False),
                               hovertemplate='Energy: %{x}<br>Popularity: %{y}'), 2, 1)
        
        # Top genres by average popularity
        if 'track_genre' in self.df.columns:
            genre_popularity = self.df.groupby('track_genre')['popularity'].mean().nlargest(10)
            fig.add_trace(go.Bar(x=genre_popularity.index, y=genre_popularity.values,
                               name='Avg Popularity', marker_color='#1DB954'), 2, 2)
        
        fig.update_layout(height=800, showlegend=False, title_text="Popularity Analysis")
        
        return fig

    def create_advanced_insights(self):
        """Create advanced analytical insights"""
        insights = {}
        
        # Insight 1: Most important features for hits
        hit_correlations = {}
        for feature in self.audio_features:
            correlation = self.df[feature].corr(self.df['popularity'])
            hit_correlations[feature] = correlation
        
        insights['feature_correlations'] = dict(sorted(hit_correlations.items(), 
                                                     key=lambda x: abs(x[1]), reverse=True))
        
        # Insight 2: Genre with highest hit rate
        if 'track_genre' in self.df.columns:
            genre_hit_rates = self.df.groupby('track_genre')['is_hit'].mean().nlargest(5)
            insights['top_genres_by_hit_rate'] = genre_hit_rates.to_dict()
        
        # Insight 3: Feature ranges for hits
        hit_data = self.df[self.df['is_hit'] == 1]
        insights['hit_feature_ranges'] = {}
        for feature in self.audio_features[:4]:  # First 4 features for brevity
            insights['hit_feature_ranges'][feature] = {
                'min': hit_data[feature].min(),
                'max': hit_data[feature].max(),
                'mean': hit_data[feature].mean()
            }
        
        return insights

    def get_statistical_summary(self):
        """Generate comprehensive statistical summary"""
        summary = {}
        
        # Basic statistics
        summary['basic'] = {
            'total_songs': len(self.df),
            'hit_songs': self.df['is_hit'].sum(),
            'hit_rate': f"{(self.df['is_hit'].sum() / len(self.df) * 100):.2f}%",
            'avg_popularity': f"{self.df['popularity'].mean():.2f}",
            'most_common_genre': self.df['track_genre'].mode()[0] if 'track_genre' in self.df.columns else 'N/A'
        }
        
        # Feature statistics
        summary['features'] = {}
        for feature in self.audio_features:
            summary['features'][feature] = {
                'mean': f"{self.df[feature].mean():.3f}",
                'std': f"{self.df[feature].std():.3f}",
                'hit_mean': f"{self.df[self.df['is_hit'] == 1][feature].mean():.3f}",
                'non_hit_mean': f"{self.df[self.df['is_hit'] == 0][feature].mean():.3f}"
            }
        
        return summary

    # ==================== TREND ANALYSIS METHODS ====================

    def create_trend_analysis_dashboard(self, selected_features=None, genre_filter=None, popularity_range=None):
        """
        Comprehensive trend analysis across popularity spectrum
        """
        if selected_features is None:
            selected_features = ['danceability', 'energy', 'valence', 'acousticness']
        
        # Create popularity bins to simulate "time" progression
        self.df['popularity_bin'] = pd.cut(self.df['popularity'], bins=10, labels=range(1, 11))
        
        # Apply filters
        filtered_df = self.df.copy()
        if genre_filter and genre_filter != 'All':
            filtered_df = filtered_df[filtered_df['track_genre'] == genre_filter]
        if popularity_range:
            filtered_df = filtered_df[(filtered_df['popularity'] >= popularity_range[0]) & 
                                    (filtered_df['popularity'] <= popularity_range[1])]
        
        # Calculate average features by popularity bin
        trend_data = filtered_df.groupby('popularity_bin')[selected_features].mean().reset_index()
        trend_data['popularity_bin'] = trend_data['popularity_bin'].astype(int) * 10  # Convert to approximate popularity
        
        # Create interactive line chart
        fig = go.Figure()
        
        colors = ['#1DB954', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for i, feature in enumerate(selected_features):
            fig.add_trace(go.Scatter(
                x=trend_data['popularity_bin'],
                y=trend_data[feature],
                mode='lines+markers',
                name=feature.title(),
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8),
                hovertemplate=f"<b>{feature.title()}</b><br>" +
                             "Popularity Range: %{x}-%{text}<br>" +
                             "Value: %{y:.3f}<extra></extra>",
                text=[f"{x+10}" for x in trend_data['popularity_bin']]
            ))
        
        fig.update_layout(
            title="Music Feature Trends Across Popularity Spectrum",
            xaxis_title="Popularity Range",
            yaxis_title="Feature Value",
            hovermode="x unified",
            height=500,
            showlegend=True
        )
        
        # Generate insights with filter context
        insights = self._generate_trend_insights(trend_data, selected_features, genre_filter, popularity_range, filtered_df)
        
        return fig, insights

    def _generate_trend_insights(self, trend_data, features, genre_filter, popularity_range, filtered_df):
        """Generate automatic insights from trend data with filter context"""
        insights = []
        
        # Add context about current filters
        if genre_filter and genre_filter != 'All':
            insights.append(f"🎵 **Analysis for {genre_filter} genre**")
        
        if popularity_range and popularity_range != (0, 100):
            insights.append(f"📊 **Popularity range: {popularity_range[0]}-{popularity_range[1]}**")
        
        insights.append("")  # Empty line for spacing
        
        # Calculate insights based on the actual filtered data
        for feature in features:
            # Calculate trend (slope) from the actual trend data
            if len(trend_data) > 1:  # Only calculate if we have multiple data points
                x = trend_data['popularity_bin'].values
                y = trend_data[feature].values
                slope = np.polyfit(x, y, 1)[0]
                
                # Get feature statistics for the filtered data
                feature_mean = filtered_df[feature].mean()
                hit_feature_mean = filtered_df[filtered_df['is_hit'] == 1][feature].mean() if len(filtered_df[filtered_df['is_hit'] == 1]) > 0 else 0
                
                if slope > 0.005:
                    trend_desc = f"📈 **{feature.title()}** increases with popularity (+{slope:.3f})"
                elif slope < -0.005:
                    trend_desc = f"📉 **{feature.title()}** decreases with popularity ({slope:.3f})"
                else:
                    trend_desc = f"➡️ **{feature.title()}** remains stable across popularity"
                
                # Add feature value context
                value_insight = f"   • Average: {feature_mean:.3f}"
                if hit_feature_mean > 0:
                    value_insight += f" | Hits: {hit_feature_mean:.3f}"
                
                insights.append(trend_desc)
                insights.append(value_insight)
            else:
                insights.append(f"📊 **{feature.title()}**: Insufficient data for trend analysis")
        
        insights.append("")  # Empty line for spacing
        
        # Add comparative insights
        if len(features) > 1:
            # Find feature with strongest trend
            strongest_trend_feature = None
            max_slope = 0
            
            for feature in features:
                if len(trend_data) > 1:
                    x = trend_data['popularity_bin'].values
                    y = trend_data[feature].values
                    slope = abs(np.polyfit(x, y, 1)[0])
                    if slope > max_slope:
                        max_slope = slope
                        strongest_trend_feature = feature
            
            if strongest_trend_feature:
                insights.append(f"🎯 **{strongest_trend_feature.title()}** shows strongest relationship with popularity")
        
        # Add hit song insights for the filtered data
        hit_count = filtered_df['is_hit'].sum()
        total_count = len(filtered_df)
        if total_count > 0:
            hit_rate = (hit_count / total_count) * 100
            insights.append(f"")
            insights.append(f"🔥 **Hit Analysis**: {hit_count}/{total_count} songs are hits ({hit_rate:.1f}% hit rate)")
            
            # Compare with overall dataset
            overall_hit_rate = (self.df['is_hit'].sum() / len(self.df)) * 100
            if hit_rate > overall_hit_rate:
                insights.append(f"   • 📈 Higher hit rate than overall ({overall_hit_rate:.1f}%)")
            elif hit_rate < overall_hit_rate:
                insights.append(f"   • 📉 Lower hit rate than overall ({overall_hit_rate:.1f}%)")
        
        return insights

    # ==================== GENRE ANALYSIS METHODS ====================

    def create_genre_comparison_dashboard(self, selected_genres=None, selected_features=None, comparison_type='radar'):
        """
        Comprehensive genre comparison with multiple visualization options
        """
        if selected_genres is None:
            # Get top 6 genres by default
            selected_genres = self.df['track_genre'].value_counts().head(6).index.tolist()
        
        if selected_features is None:
            selected_features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'liveness']
        
        # Filter data for selected genres
        genre_data = self.df[self.df['track_genre'].isin(selected_genres)]
        
        # Calculate statistics for each genre - SIMPLIFIED APPROACH
        genre_stats_list = []
        
        for genre in selected_genres:
            genre_df = genre_data[genre_data['track_genre'] == genre]
            if len(genre_df) > 0:
                stats = {'track_genre': genre}
                
                # Add feature means
                for feature in selected_features:
                    stats[feature] = genre_df[feature].mean()
                
                # Add additional statistics
                stats['popularity_mean'] = genre_df['popularity'].mean()
                stats['is_hit_mean'] = genre_df['is_hit'].mean()
                stats['song_count'] = len(genre_df)
                
                genre_stats_list.append(stats)
        
        # Create DataFrame from the list
        genre_stats = pd.DataFrame(genre_stats_list)
        
        if comparison_type == 'radar':
            fig = self._create_radar_chart(genre_stats, selected_genres, selected_features)
        elif comparison_type == 'heatmap':
            fig = self._create_genre_heatmap(genre_stats, selected_genres, selected_features)
        elif comparison_type == 'bar':
            fig = self._create_genre_bar_chart(genre_stats, selected_genres, selected_features)
        
        # Generate genre insights
        insights = self._generate_genre_insights(genre_stats, selected_genres, selected_features, genre_data)
        
        return fig, insights, genre_stats

    def _create_radar_chart(self, genre_stats, genres, features):
        """Create interactive radar chart for genre comparison"""
        fig = go.Figure()
        
        colors = ['#1DB954', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#FFD93D', '#6B5B95', '#88B04B']
        
        for i, genre in enumerate(genres):
            genre_values = genre_stats[genre_stats['track_genre'] == genre]
            if len(genre_values) > 0:
                values = []
                for feature in features:
                    if feature in genre_values.columns:
                        values.append(genre_values[feature].values[0])
                    else:
                        values.append(0)  # Default value if column missing
                
                # Close the radar chart
                values = values + [values[0]]
                radar_features = features + [features[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=radar_features,
                    fill='toself',
                    name=genre,
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.7
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Genre Feature Comparison (Radar Chart)",
            height=600
        )
        
        return fig

    def _create_genre_heatmap(self, genre_stats, genres, features):
        """Create heatmap showing feature values across genres"""
        # Prepare data for heatmap
        heatmap_data = []
        for genre in genres:
            genre_values = genre_stats[genre_stats['track_genre'] == genre]
            if len(genre_values) > 0:
                row = []
                for feature in features:
                    if feature in genre_values.columns:
                        row.append(genre_values[feature].values[0])
                    else:
                        row.append(0)  # Default value
                heatmap_data.append(row)
        
        fig = px.imshow(heatmap_data,
                       x=features,
                       y=genres,
                       color_continuous_scale='Viridis',
                       aspect="auto")
        
        fig.update_layout(
            title="Genre Feature Heatmap",
            xaxis_title="Features",
            yaxis_title="Genres",
            height=500
        )
        
        # Add annotations
        for i in range(len(genres)):
            for j in range(len(features)):
                fig.add_annotation(x=j, y=i, 
                                 text=f"{heatmap_data[i][j]:.2f}",
                                 showarrow=False,
                                 font=dict(color="white" if heatmap_data[i][j] > 0.5 else "black"))
        
        return fig

    def _create_genre_bar_chart(self, genre_stats, genres, features):
        """Create grouped bar chart for genre comparison"""
        # Prepare data for bar chart
        bar_data = []
        for genre in genres:
            genre_values = genre_stats[genre_stats['track_genre'] == genre]
            if len(genre_values) > 0:
                for feature in features:
                    if feature in genre_values.columns:
                        bar_data.append({
                            'Genre': genre,
                            'Feature': feature,
                            'Value': genre_values[feature].values[0]
                        })
        
        bar_df = pd.DataFrame(bar_data)
        
        fig = px.bar(bar_df, x='Genre', y='Value', color='Feature',
                    title="Genre Feature Comparison (Bar Chart)",
                    barmode='group',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_layout(height=500, xaxis_tickangle=-45)
        return fig

    def _generate_genre_insights(self, genre_stats, genres, features, genre_data):
        """Generate dynamic insights for genre comparison"""
        insights = []
        
        insights.append("🎵 **Genre Comparison Insights**")
        insights.append("")
        
        # Find genre with highest hit rate
        if 'is_hit_mean' in genre_stats.columns:
            best_hit_idx = genre_stats['is_hit_mean'].idxmax()
            worst_hit_idx = genre_stats['is_hit_mean'].idxmin()
            
            best_genre = genre_stats.iloc[best_hit_idx]
            worst_genre = genre_stats.iloc[worst_hit_idx]
            
            insights.append(f"🏆 **Highest Hit Rate**: {best_genre['track_genre']} ({best_genre['is_hit_mean']*100:.1f}%)")
            insights.append(f"📉 **Lowest Hit Rate**: {worst_genre['track_genre']} ({worst_genre['is_hit_mean']*100:.1f}%)")
            insights.append("")
        
        # Analyze feature extremes
        for feature in features[:4]:  # First 4 features for brevity
            if feature in genre_stats.columns:
                max_idx = genre_stats[feature].idxmax()
                min_idx = genre_stats[feature].idxmin()
                
                max_genre = genre_stats.iloc[max_idx]
                min_genre = genre_stats.iloc[min_idx]
                
                insights.append(f"📊 **{feature.title()}**:")
                insights.append(f"   • Highest: {max_genre['track_genre']} ({max_genre[feature]:.3f})")
                insights.append(f"   • Lowest: {min_genre['track_genre']} ({min_genre[feature]:.3f})")
        
        insights.append("")
        
        # Genre characteristics
        insights.append("🎯 **Genre Profiles**:")
        for genre in genres[:4]:  # First 4 genres for brevity
            genre_info = genre_stats[genre_stats['track_genre'] == genre]
            if len(genre_info) > 0:
                genre_info = genre_info.iloc[0]
                top_features = []
                
                for feature in features:
                    if feature in genre_info.index and genre_info[feature] > 0.6:  # High values
                        top_features.append(feature)
                
                if top_features:
                    insights.append(f"   • **{genre}**: High {', '.join(top_features[:3])}")
        
        return insights

    # ==================== CORRELATION EXPLORER METHODS ====================

    def create_correlation_explorer(self, selected_features=None, color_by='is_hit', genre_filter=None, popularity_range=None):
        """
        Advanced correlation analysis with multiple visualization options
        """
        if selected_features is None:
            selected_features = ['danceability', 'energy', 'valence', 'acousticness', 'popularity']
        
        # Apply filters
        filtered_df = self.df.copy()
        if genre_filter and genre_filter != 'All':
            filtered_df = filtered_df[filtered_df['track_genre'] == genre_filter]
        if popularity_range:
            filtered_df = filtered_df[(filtered_df['popularity'] >= popularity_range[0]) & 
                                    (filtered_df['popularity'] <= popularity_range[1])]
        
        # Calculate correlation matrix
        corr_matrix = filtered_df[selected_features].corr()
        
        # Create multiple visualization options
        scatter_matrix_fig = self._create_scatter_matrix(filtered_df, selected_features, color_by)
        correlation_heatmap_fig = self._create_advanced_correlation_heatmap(corr_matrix, selected_features)
        feature_importance_fig = self._create_feature_importance_chart(corr_matrix, selected_features)
        
        # Generate correlation insights
        insights = self._generate_correlation_insights(corr_matrix, selected_features, filtered_df)
        
        return scatter_matrix_fig, correlation_heatmap_fig, feature_importance_fig, insights

    def _create_scatter_matrix(self, data, features, color_by):
        """Create interactive scatter plot matrix"""
        if color_by == 'is_hit':
            color_col = 'is_hit'
            color_discrete_map = {0: '#FF4B4B', 1: '#1DB954'}
        elif color_by == 'track_genre':
            color_col = 'track_genre'
            color_discrete_map = None
        else:
            color_col = None
            color_discrete_map = None
        
        fig = px.scatter_matrix(data, 
                              dimensions=features,
                              color=color_col,
                              title="Feature Relationships - Scatter Plot Matrix",
                              opacity=0.6,
                              color_discrete_map=color_discrete_map,
                              hover_data=['track_name', 'artists'] if 'track_name' in data.columns else None)
        
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(height=800)
        
        return fig

    def _create_advanced_correlation_heatmap(self, corr_matrix, features):
        """Create enhanced correlation heatmap with annotations"""
        fig = px.imshow(corr_matrix,
                       x=features,
                       y=features,
                       color_continuous_scale='RdBu_r',
                       aspect="auto",
                       zmin=-1, zmax=1,
                       title="Feature Correlation Heatmap")
        
        # Add correlation values as annotations
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                correlation_value = corr_matrix.iloc[i, j]
                color = "white" if abs(correlation_value) > 0.5 else "black"
                fig.add_annotation(x=j, y=i, 
                                 text=f"{correlation_value:.2f}",
                                 showarrow=False,
                                 font=dict(color=color, size=10))
        
        fig.update_layout(height=500)
        return fig

    def _create_feature_importance_chart(self, corr_matrix, features):
        """Create feature importance chart based on correlation with popularity"""
        if 'popularity' in features:
            # Get correlations with popularity
            pop_correlations = corr_matrix['popularity'].abs().sort_values(ascending=False)
            # Remove popularity itself
            pop_correlations = pop_correlations[pop_correlations.index != 'popularity']
            
            importance_df = pd.DataFrame({
                'feature': pop_correlations.index,
                'correlation': pop_correlations.values
            })
            
            fig = px.bar(importance_df, 
                        x='correlation', 
                        y='feature',
                        orientation='h',
                        title="Feature Importance for Popularity (Absolute Correlation)",
                        color='correlation',
                        color_continuous_scale='Viridis')
            
            fig.update_layout(height=400, showlegend=False)
            return fig
        else:
            # If popularity not selected, show general feature relationships
            avg_correlations = corr_matrix.abs().mean().sort_values(ascending=False)
            importance_df = pd.DataFrame({
                'feature': avg_correlations.index,
                'avg_correlation': avg_correlations.values
            })
            
            fig = px.bar(importance_df, 
                        x='avg_correlation', 
                        y='feature',
                        orientation='h',
                        title="Feature Connectivity (Average Absolute Correlation)",
                        color='avg_correlation',
                        color_continuous_scale='Viridis')
            
            fig.update_layout(height=400, showlegend=False)
            return fig

    def _generate_correlation_insights(self, corr_matrix, features, filtered_df):
        """Generate dynamic correlation insights"""
        insights = []
        
        insights.append("🔗 **Correlation Analysis Insights**")
        insights.append("")
        
        # Find strongest correlations
        strong_correlations = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                feature1 = corr_matrix.columns[i]
                feature2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                
                if abs(correlation) > 0.3:  # Moderate to strong correlation
                    direction = "positive" if correlation > 0 else "negative"
                    strong_correlations.append((feature1, feature2, correlation, direction))
        
        # Sort by absolute correlation strength
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        insights.append("💪 **Strongest Relationships:**")
        for feature1, feature2, corr, direction in strong_correlations[:5]:
            insights.append(f"   • {feature1.title()} ↔ {feature2.title()}: {corr:.2f} ({direction})")
        
        insights.append("")
        
        # Popularity correlations if available
        if 'popularity' in features:
            pop_correlations = corr_matrix['popularity'].sort_values(ascending=False)
            # Remove popularity itself
            pop_correlations = pop_correlations[pop_correlations.index != 'popularity']
            
            insights.append("📈 **Popularity Drivers:**")
            for feature, corr in pop_correlations.head(3).items():
                insights.append(f"   • {feature.title()}: {corr:.2f}")
            
            insights.append("")
            insights.append("📉 **Popularity Limiters:**")
            for feature, corr in pop_correlations.tail(3).items():
                insights.append(f"   • {feature.title()}: {corr:.2f}")
        
        insights.append("")
        
        # Statistical context
        insights.append("📊 **Statistical Context:**")
        insights.append(f"   • Songs analyzed: {len(filtered_df):,}")
        if 'track_genre' in filtered_df.columns and filtered_df['track_genre'].nunique() > 1:
            insights.append(f"   • Genres included: {filtered_df['track_genre'].nunique()}")
        if 'is_hit' in filtered_df.columns:
            hit_rate = (filtered_df['is_hit'].sum() / len(filtered_df)) * 100
            insights.append(f"   • Hit rate: {hit_rate:.1f}%")
        
        return insights

    def create_3d_correlation_plot(self, x_feature, y_feature, z_feature, color_by='is_hit', genre_filter=None):
        """Create interactive 3D scatter plot"""
        filtered_df = self.df.copy()
        if genre_filter and genre_filter != 'All':
            filtered_df = filtered_df[filtered_df['track_genre'] == genre_filter]
        
        if color_by == 'is_hit':
            color_col = 'is_hit'
            color_discrete_map = {0: '#FF4B4B', 1: '#1DB954'}
        elif color_by == 'track_genre':
            color_col = 'track_genre'
            color_discrete_map = None
        else:
            color_col = None
            color_discrete_map = None
        
        fig = px.scatter_3d(filtered_df, 
                           x=x_feature,
                           y=y_feature, 
                           z=z_feature,
                           color=color_col,
                           title=f"3D Feature Space: {x_feature} vs {y_feature} vs {z_feature}",
                           color_discrete_map=color_discrete_map,
                           opacity=0.6,
                           hover_data=['track_name', 'artists', 'popularity'] if 'track_name' in filtered_df.columns else None)
        
        fig.update_layout(height=600)
        return fig

    # ==================== HIT SONG PROFILER METHODS ====================

    def create_hit_song_profiler(self, selected_features=None, genre_filter=None, comparison_type='radar'):
        """
        Comprehensive analysis of what makes hit songs different
        """
        if selected_features is None:
            selected_features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'liveness']
        
        # Apply genre filter
        filtered_df = self.df.copy()
        if genre_filter and genre_filter != 'All':
            filtered_df = filtered_df[filtered_df['track_genre'] == genre_filter]
        
        # Separate hit and non-hit songs
        hit_songs = filtered_df[filtered_df['is_hit'] == 1]
        non_hit_songs = filtered_df[filtered_df['is_hit'] == 0]
        
        if len(hit_songs) == 0:
            raise ValueError("No hit songs found in the selected data. Try adjusting filters.")
        
        # Calculate statistics
        hit_stats = hit_songs[selected_features].mean()
        non_hit_stats = non_hit_songs[selected_features].mean()
        
        # Calculate percentage differences
        differences = ((hit_stats - non_hit_stats) / non_hit_stats * 100).fillna(0)
        
        if comparison_type == 'radar':
            fig = self._create_hit_radar_chart(hit_stats, non_hit_stats, selected_features)
        elif comparison_type == 'bar':
            fig = self._create_hit_bar_chart(hit_stats, non_hit_stats, selected_features, differences)
        elif comparison_type == 'violin':
            fig = self._create_hit_violin_plot(filtered_df, selected_features)
        
        # Generate hit song insights
        insights = self._generate_hit_song_insights(hit_songs, non_hit_songs, selected_features, differences, genre_filter)
        
        return fig, insights, hit_stats, non_hit_stats, differences

    def _create_hit_radar_chart(self, hit_stats, non_hit_stats, features):
        """Create radar chart comparing hit vs non-hit profiles"""
        fig = go.Figure()
        
        # Hit songs
        hit_values = hit_stats.values.tolist() + [hit_stats.values[0]]
        radar_features = features + [features[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=hit_values,
            theta=radar_features,
            fill='toself',
            name='Hit Songs',
            line=dict(color='#1DB954', width=3),
            opacity=0.8
        ))
        
        # Non-hit songs
        non_hit_values = non_hit_stats.values.tolist() + [non_hit_stats.values[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=non_hit_values,
            theta=radar_features,
            fill='toself',
            name='Non-Hit Songs',
            line=dict(color='#FF4B4B', width=3),
            opacity=0.6
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Hit vs Non-Hit Song Profile Comparison",
            height=600
        )
        
        return fig

    def _create_hit_bar_chart(self, hit_stats, non_hit_stats, features, differences):
        """Create bar chart showing feature differences"""
        comparison_data = []
        for feature in features:
            comparison_data.extend([
                {'Feature': feature, 'Value': hit_stats[feature], 'Type': 'Hit Songs'},
                {'Feature': feature, 'Value': non_hit_stats[feature], 'Type': 'Non-Hit Songs'}
            ])
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig = px.bar(comparison_df, 
                    x='Feature', 
                    y='Value', 
                    color='Type',
                    barmode='group',
                    title="Feature Comparison: Hit vs Non-Hit Songs",
                    color_discrete_map={'Hit Songs': '#1DB954', 'Non-Hit Songs': '#FF4B4B'})
        
        # Add difference annotations
        for i, feature in enumerate(features):
            diff = differences[feature]
            if abs(diff) > 5:  # Only show significant differences
                fig.add_annotation(
                    x=i,
                    y=max(hit_stats[feature], non_hit_stats[feature]) + 0.05,
                    text=f"{diff:+.1f}%",
                    showarrow=False,
                    font=dict(color="black", size=10)
                )
        
        fig.update_layout(height=500)
        return fig

    def _create_hit_violin_plot(self, data, features):
        """Create violin plots showing distribution differences"""
        # Melt data for violin plot
        melted_data = data.melt(id_vars=['is_hit'], 
                               value_vars=features, 
                               var_name='Feature', 
                               value_name='Value')
        
        fig = px.violin(melted_data, 
                       x='Feature', 
                       y='Value', 
                       color='is_hit',
                       box=True,
                       points=False,
                       title="Feature Distributions: Hit vs Non-Hit Songs",
                       color_discrete_map={0: '#FF4B4B', 1: '#1DB954'})
        
        fig.update_layout(height=500, xaxis_tickangle=-45)
        return fig

    def _generate_hit_song_insights(self, hit_songs, non_hit_songs, features, differences, genre_filter):
        """Generate insights about what makes hit songs successful"""
        insights = []
        
        insights.append("🎵 **Hit Song Analysis**")
        if genre_filter and genre_filter != 'All':
            insights.append(f"**Genre**: {genre_filter}")
        insights.append(f"**Hit Songs Analyzed**: {len(hit_songs):,}")
        insights.append(f"**Non-Hit Songs Analyzed**: {len(non_hit_songs):,}")
        insights.append("")
        
        # Find most different features
        significant_differences = differences[abs(differences) > 5].sort_values(ascending=False)
        
        if len(significant_differences) > 0:
            insights.append("💪 **Key Differentiators:**")
            for feature, diff in significant_differences.head(5).items():
                direction = "higher" if diff > 0 else "lower"
                hit_value = hit_songs[feature].mean()
                non_hit_value = non_hit_songs[feature].mean()
                insights.append(f"   • **{feature.title()}**: {direction} by {abs(diff):.1f}%")
                insights.append(f"     (Hits: {hit_value:.3f} vs Non-hits: {non_hit_value:.3f})")
        else:
            insights.append("📊 **No strong differentiators found** - hit songs are similar to non-hits in selected features")
        
        insights.append("")
        
        # Hit song characteristics
        insights.append("🎯 **Hit Song Profile:**")
        for feature in features[:4]:  # Show first 4 features
            hit_avg = hit_songs[feature].mean()
            hit_std = hit_songs[feature].std()
            insights.append(f"   • **{feature.title()}**: {hit_avg:.3f} ± {hit_std:.3f}")
        
        insights.append("")
        
        # Statistical significance
        insights.append("📈 **Statistical Context:**")
        hit_rate = (len(hit_songs) / (len(hit_songs) + len(non_hit_songs))) * 100
        insights.append(f"   • **Hit Rate**: {hit_rate:.1f}%")
        
        # Compare with overall dataset if filtered
        if genre_filter and genre_filter != 'All':
            overall_hit_rate = (self.df['is_hit'].sum() / len(self.df)) * 100
            if hit_rate > overall_hit_rate:
                insights.append(f"   • 📈 **Higher hit rate** than overall ({overall_hit_rate:.1f}%)")
            else:
                insights.append(f"   • 📉 **Lower hit rate** than overall ({overall_hit_rate:.1f}%)")
        
        return insights

    def create_feature_threshold_analysis(self, feature, genre_filter=None):
        """Analyze hit rate across different feature thresholds"""
        filtered_df = self.df.copy()
        if genre_filter and genre_filter != 'All':
            filtered_df = filtered_df[filtered_df['track_genre'] == genre_filter]
        
        # Create bins for the feature
        filtered_df['feature_bin'] = pd.cut(filtered_df[feature], bins=10)
        
        # Calculate hit rate for each bin
        bin_analysis = filtered_df.groupby('feature_bin').agg({
            'is_hit': 'mean',
            feature: 'mean',
            'track_id': 'count'
        }).reset_index()
        
        bin_analysis['hit_rate'] = bin_analysis['is_hit'] * 100
        bin_analysis = bin_analysis.rename(columns={'track_id': 'song_count'})
        
        # Create visualization
        fig = px.line(bin_analysis, 
                     x=feature, 
                     y='hit_rate',
                     title=f"Hit Rate vs {feature.title()}",
                     markers=True)
        
        fig.update_layout(
            xaxis_title=f"{feature.title()}",
            yaxis_title="Hit Rate (%)",
            height=400
        )
        
        # Add song count as size of markers
        fig.update_traces(marker=dict(size=bin_analysis['song_count']/bin_analysis['song_count'].max()*20+5))
        
        return fig, bin_analysis

    # ==================== DISTRIBUTION ANALYZER METHODS ====================

    def create_distribution_analyzer(self, selected_features=None, genre_filter=None, distribution_type='histogram', compare_hits=True):
        """
        Comprehensive feature distribution analysis with statistical insights
        """
        if selected_features is None:
            selected_features = ['danceability', 'energy', 'valence', 'acousticness']
        
        # Apply filters
        filtered_df = self.df.copy()
        if genre_filter and genre_filter != 'All':
            filtered_df = filtered_df[filtered_df['track_genre'] == genre_filter]
        
        # Separate data if comparing hits
        if compare_hits:
            hit_data = filtered_df[filtered_df['is_hit'] == 1]
            non_hit_data = filtered_df[filtered_df['is_hit'] == 0]
        else:
            hit_data = filtered_df
            non_hit_data = None
        
        if distribution_type == 'histogram':
            fig = self._create_advanced_histogram(filtered_df, selected_features, compare_hits)
        elif distribution_type == 'density':
            fig = self._create_density_plot(filtered_df, selected_features, compare_hits)
        elif distribution_type == 'box':
            fig = self._create_box_plot(filtered_df, selected_features, compare_hits)
        elif distribution_type == 'qq':
            fig = self._create_qq_plot(filtered_df, selected_features)
        elif distribution_type == 'ecdf':
            fig = self._create_ecdf_plot(filtered_df, selected_features, compare_hits)
        
        # Generate distribution insights
        insights = self._generate_distribution_insights(filtered_df, selected_features, hit_data, non_hit_data, genre_filter)
        
        return fig, insights

    def _create_advanced_histogram(self, data, features, compare_hits):
        """Create interactive histogram with multiple features"""
        if compare_hits:
            # Create subplots for hit vs non-hit comparison
            fig = make_subplots(rows=len(features), cols=2, 
                              subplot_titles=[f"{feature.title()} - Distribution" for feature in features] * 2,
                              horizontal_spacing=0.05, vertical_spacing=0.06)
            
            for i, feature in enumerate(features):
                row = i + 1
                
                # Hit songs histogram
                hit_data = data[data['is_hit'] == 1][feature]
                fig.add_trace(go.Histogram(x=hit_data, name='Hit Songs', 
                                         marker_color='#1DB954', opacity=0.7,
                                         nbinsx=30, showlegend=(i==0)),
                            row=row, col=1)
                
                # Non-hit songs histogram
                non_hit_data = data[data['is_hit'] == 0][feature]
                fig.add_trace(go.Histogram(x=non_hit_data, name='Non-Hit Songs', 
                                         marker_color='#FF4B4B', opacity=0.7,
                                         nbinsx=30, showlegend=(i==0)),
                            row=row, col=2)
            
            fig.update_layout(height=300 * len(features), 
                             title_text="Feature Distributions: Hit vs Non-Hit Songs",
                             barmode='overlay')
        else:
            # Single histogram for all data
            fig = px.histogram(data, x=features, 
                              title="Feature Distributions",
                              opacity=0.7,
                              nbins=30,
                              color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=400)
        
        return fig

    def _create_density_plot(self, data, features, compare_hits):
        """Create density plots (KDE) for feature distributions"""
        if compare_hits:
            fig = go.Figure()
            
            for feature in features:
                # Hit songs density
                hit_data = data[data['is_hit'] == 1][feature].dropna()
                if len(hit_data) > 1:
                    fig.add_trace(go.Violin(x=[feature] * len(hit_data), y=hit_data,
                                          name=f'{feature} - Hits', side='positive',
                                          line_color='#1DB954', opacity=0.7))
                
                # Non-hit songs density
                non_hit_data = data[data['is_hit'] == 0][feature].dropna()
                if len(non_hit_data) > 1:
                    fig.add_trace(go.Violin(x=[feature] * len(non_hit_data), y=non_hit_data,
                                          name=f'{feature} - Non-Hits', side='negative',
                                          line_color='#FF4B4B', opacity=0.7))
            
            fig.update_layout(title="Feature Density Distributions: Hit vs Non-Hit Songs",
                             violinmode='overlay',
                             height=500)
        else:
            fig = px.violin(data, y=features, box=True, points=False,
                           title="Feature Density Distributions",
                           color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=500)
        
        return fig

    def _create_box_plot(self, data, features, compare_hits):
        """Create box plots for feature distributions"""
        if compare_hits:
            # Melt data for box plot
            melted_data = data.melt(id_vars=['is_hit'], 
                                   value_vars=features, 
                                   var_name='Feature', 
                                   value_name='Value')
            
            fig = px.box(melted_data, 
                        x='Feature', 
                        y='Value', 
                        color='is_hit',
                        title="Feature Distributions: Box Plot Comparison",
                        color_discrete_map={0: '#FF4B4B', 1: '#1DB954'})
        else:
            fig = px.box(data, y=features, 
                        title="Feature Distributions: Box Plot",
                        color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_layout(height=500, xaxis_tickangle=-45)
        return fig

    def _create_qq_plot(self, data, features):
        """Create Q-Q plots to check normality"""
        fig = make_subplots(rows=len(features), cols=1,
                           subplot_titles=[f"Q-Q Plot: {feature.title()}" for feature in features],
                           vertical_spacing=0.08)
        
        for i, feature in enumerate(features):
            row = i + 1
            feature_data = data[feature].dropna()
            
            # Generate theoretical quantiles
            theoretical_quantiles = stats.probplot(feature_data, dist="norm")
            x = theoretical_quantiles[0][0]
            y = theoretical_quantiles[0][1]
            
            # Add Q-Q line
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', 
                                   name=feature, marker=dict(size=4)),
                        row=row, col=1)
            
            # Add reference line
            line_x = [x.min(), x.max()]
            line_y = [theoretical_quantiles[1][1] + theoretical_quantiles[1][0] * line_x[0],
                     theoretical_quantiles[1][1] + theoretical_quantiles[1][0] * line_x[1]]
            
            fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines',
                                   line=dict(color='red', dash='dash'),
                                   showlegend=False),
                        row=row, col=1)
        
        fig.update_layout(height=300 * len(features), 
                         title_text="Q-Q Plots: Normality Check",
                         showlegend=False)
        return fig

    def _create_ecdf_plot(self, data, features, compare_hits):
        """Create Empirical Cumulative Distribution Function plots"""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, feature in enumerate(features):
            if compare_hits:
                # Hit songs ECDF
                hit_data = data[data['is_hit'] == 1][feature].dropna().sort_values()
                hit_ecdf = np.arange(1, len(hit_data) + 1) / len(hit_data)
                fig.add_trace(go.Scatter(x=hit_data, y=hit_ecdf,
                                       mode='lines', name=f'{feature} - Hits',
                                       line=dict(color=colors[i % len(colors)], width=2)))
                
                # Non-hit songs ECDF
                non_hit_data = data[data['is_hit'] == 0][feature].dropna().sort_values()
                non_hit_ecdf = np.arange(1, len(non_hit_data) + 1) / len(non_hit_data)
                fig.add_trace(go.Scatter(x=non_hit_data, y=non_hit_ecdf,
                                       mode='lines', name=f'{feature} - Non-Hits',
                                       line=dict(color=colors[i % len(colors)], width=2, dash='dash')))
            else:
                # All data ECDF
                feature_data = data[feature].dropna().sort_values()
                ecdf = np.arange(1, len(feature_data) + 1) / len(feature_data)
                fig.add_trace(go.Scatter(x=feature_data, y=ecdf,
                                       mode='lines', name=feature,
                                       line=dict(color=colors[i % len(colors)], width=2)))
        
        fig.update_layout(title="Empirical Cumulative Distribution Functions",
                         xaxis_title="Feature Value",
                         yaxis_title="Cumulative Probability",
                         height=500)
        return fig

    def _generate_distribution_insights(self, data, features, hit_data, non_hit_data, genre_filter):
        """Generate statistical insights about feature distributions"""
        insights = []
        
        insights.append("📊 **Distribution Analysis**")
        if genre_filter and genre_filter != 'All':
            insights.append(f"**Genre**: {genre_filter}")
        insights.append(f"**Total Songs**: {len(data):,}")
        if hit_data is not None and non_hit_data is not None:
            insights.append(f"**Hit Songs**: {len(hit_data):,}")
            insights.append(f"**Non-Hit Songs**: {len(non_hit_data):,}")
        insights.append("")
        
        # Statistical moments for each feature
        insights.append("📈 **Statistical Moments:**")
        for feature in features[:4]:  # Show first 4 features for brevity
            feature_data = data[feature].dropna()
            
            mean = feature_data.mean()
            std = feature_data.std()
            skew = feature_data.skew()
            kurtosis = feature_data.kurtosis()
            
            insights.append(f"**{feature.title()}:**")
            insights.append(f"   • Mean: {mean:.3f}")
            insights.append(f"   • Std Dev: {std:.3f}")
            insights.append(f"   • Skewness: {skew:.2f} {'(right)' if skew > 0.5 else '(left)' if skew < -0.5 else '(symmetric)'}")
            insights.append(f"   • Kurtosis: {kurtosis:.2f} {'(heavy-tailed)' if kurtosis > 1 else '(light-tailed)' if kurtosis < -1 else '(normal)'}")
        
        insights.append("")
        
        # Distribution comparisons
        if hit_data is not None and non_hit_data is not None:
            insights.append("🔍 **Distribution Differences:**")
            for feature in features[:3]:
                hit_mean = hit_data[feature].mean()
                non_hit_mean = non_hit_data[feature].mean()
                hit_std = hit_data[feature].std()
                non_hit_std = non_hit_data[feature].std()
                
                mean_diff = hit_mean - non_hit_mean
                std_ratio = hit_std / non_hit_std
                
                insights.append(f"**{feature.title()}:**")
                insights.append(f"   • Mean difference: {mean_diff:+.3f}")
                insights.append(f"   • Std dev ratio: {std_ratio:.2f}")
                
                if abs(mean_diff) > 0.05:
                    direction = "higher" if mean_diff > 0 else "lower"
                    insights.append(f"   • Hits have {direction} {feature}")
        
        insights.append("")
        
        # Practical insights
        insights.append("🎯 **Practical Insights:**")
        for feature in features[:2]:
            q25 = data[feature].quantile(0.25)
            q75 = data[feature].quantile(0.75)
            iqr = q75 - q25
            
            insights.append(f"**{feature.title()}:**")
            insights.append(f"   • IQR: {q25:.3f} - {q75:.3f}")
            insights.append(f"   • Middle 50% range: {iqr:.3f}")
            
            if hit_data is not None:
                hit_median = hit_data[feature].median()
                insights.append(f"   • Hit song median: {hit_median:.3f}")
        
        return insights


# Test function
def test_analyzer():
    """Test the DataAnalyzer class"""
    import os
    
    # Find CSV files in current directory
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV files found in directory")
        return
    
    print(f"📁 Found CSV file: {csv_files[0]}")
    analyzer = DataAnalyzer(csv_files[0])
    
    if hasattr(analyzer, 'df') and analyzer.df is not None:
        print("✅ DataAnalyzer initialized successfully!")
        
        # Test basic info
        info = analyzer.get_basic_info()
        print(f"📊 Basic Info: {info}")
        
        # Test statistical summary
        summary = analyzer.get_statistical_summary()
        print(f"📈 Statistical Summary Generated")
        
        # Test insights
        insights = analyzer.create_advanced_insights()
        print(f"🔍 Advanced Insights Generated")
        
        print("🎉 All tests passed! Ready to use in Streamlit app.")
    else:
        print("❌ Failed to initialize DataAnalyzer")

if __name__ == "__main__":
    test_analyzer()