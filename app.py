import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_analysis import DataAnalyzer
import os
import numpy as np
from prediction_interface import PredictionInterface

# Page configuration
st.set_page_config(
    page_title="Spotify Hit Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1DB954;
        border-bottom: 2px solid #1DB954;
        padding-bottom: 0.5rem;
    }
    .feature-card {
        background-color: #5D5E60;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Initialize data analyzer
@st.cache_data
def load_data():
    try:
        # Try different possible dataset names
        dataset_files = [f for f in os.listdir() if f.endswith('.csv')]
        if dataset_files:
            analyzer = DataAnalyzer(dataset_files[0])
            return analyzer
        else:
            st.error("No CSV dataset found in the directory!")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Spotify Hit Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "🏠 Dashboard Overview", 
        "📈 Trend Analysis", 
        "🎯 Genre Analysis",
        "📊 Correlation Explorer", 
        "🎯 Hit Song Profiler",
        "📈 Distribution Analyzer",
        "🤖 AI Prediction"
    ])

    
    # Load data
    analyzer = load_data()
    
    if analyzer is None:
        st.warning("Please make sure your dataset file is in the same folder as this app.")
        return
    
     # Initialize prediction interface
    prediction_interface = PredictionInterface(analyzer)
    
    # Page routing - ADD THIS NEW PAGE
    if page == "🏠 Dashboard Overview":
        show_dashboard(analyzer)
    elif page == "📈 Trend Analysis":
        show_trend_analysis(analyzer)
    elif page == "🎯 Genre Analysis":
        show_genre_analysis(analyzer)
    elif page == "📊 Correlation Explorer":
        show_feature_insights(analyzer)
    elif page == "📈 Distribution Analyzer":
    
        show_distribution_analyzer(analyzer)
    elif page == "🎯 Hit Song Profiler":  # ADD THIS
        show_hit_song_profiler(analyzer)
    elif page == "🤖 AI Prediction":  # NEW PAGE
        prediction_interface.show_prediction_dashboard()


def show_dashboard(analyzer):
    st.markdown('<h2 class="section-header">📊 Project Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics - using actual columns from your dataset
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Songs", f"{len(analyzer.df):,}")
    with col2:
        st.metric("Hit Songs", f"{analyzer.df['is_hit'].sum():,}")
    with col3:
        st.metric("Average Popularity", f"{analyzer.df['popularity'].mean():.1f}")
    with col4:
        unique_genres = analyzer.df['track_genre'].nunique()
        st.metric("Genres", unique_genres)
    
    # Quick insights
    st.markdown("---")
    st.markdown('<h3 class="section-header">🚀 Quick Insights</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h4>🎯 For Composers</h4>
        <ul>
        <li>Discover what makes songs popular</li>
        <li>Compare your music to hit songs</li>
        <li>Understand genre-specific patterns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <h4>📈 For Analysts</h4>
        <ul>
        <li>Explore music feature relationships</li>
        <li>Analyze feature correlations</li>
        <li>Study genre characteristics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Basic statistics
        st.markdown("""
        <div class="feature-card">
        <h4>📊 Quick Stats</h4>
        <ul>
        <li>Hit Rate: {:.2f}%</li>
        <li>Most Common Genre: {}</li>
        <li>Avg Danceability: {:.3f}</li>
        <li>Avg Energy: {:.3f}</li>
        </ul>
        </div>
        """.format(
            (analyzer.df['is_hit'].sum() / len(analyzer.df) * 100),
            analyzer.df['track_genre'].mode()[0] if 'track_genre' in analyzer.df.columns else 'N/A',
            analyzer.df['danceability'].mean(),
            analyzer.df['energy'].mean()
        ), unsafe_allow_html=True)
    
    with col2:
        # Quick feature distribution
        fig = px.histogram(analyzer.df, x='danceability', 
                          title="Danceability Distribution",
                          color_discrete_sequence=['#1DB954'])
        st.plotly_chart(fig, use_container_width=True)

        # Hit vs Non-hit comparison
        hit_count = analyzer.df['is_hit'].sum()
        non_hit_count = len(analyzer.df) - hit_count
        fig_pie = px.pie(values=[hit_count, non_hit_count], 
                        names=['Hit Songs', 'Non-Hit Songs'],
                        title="Hit vs Non-Hit Distribution",
                        color_discrete_sequence=['#1DB954', '#FF4B4B'])
        st.plotly_chart(fig_pie, use_container_width=True)

def show_trend_analysis(analyzer):
    st.markdown('<h2 class="section-header">📈 Advanced Trend Analysis</h2>', unsafe_allow_html=True)
    
    st.info("🎵 **Analyze how music features change across popularity ranges** - Discover what makes songs popular!")
    
    # Filters Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Feature selection
        available_features = ['danceability', 'energy', 'valence', 'acousticness', 
                            'speechiness', 'instrumentalness', 'liveness', 'tempo']
        selected_features = st.multiselect(
            "Select features to analyze:",
            available_features,
            default=['danceability', 'energy', 'valence']
        )
    
    with col2:
        # Genre filter
        all_genres = ['All'] + analyzer.df['track_genre'].value_counts().head(15).index.tolist()
        selected_genre = st.selectbox("Filter by genre:", all_genres)
    
    with col3:
        # Popularity range
        min_pop, max_pop = st.slider(
            "Popularity range:",
            min_value=0,
            max_value=100,
            value=(0, 100)
        )
    
    if not selected_features:
        st.warning("Please select at least one feature to analyze.")
        return
    
    # Generate the trend analysis
    fig, insights = analyzer.create_trend_analysis_dashboard(
        selected_features=selected_features,
        genre_filter=selected_genre if selected_genre != 'All' else None,
        popularity_range=(min_pop, max_pop)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Display insights
    st.markdown("---")
    st.markdown('<h3 class="section-header">🔍 Key Insights</h3>', unsafe_allow_html=True)
    
    for insight in insights:
        st.write(insight)
    
    # Statistical Summary
    st.markdown("---")
    st.markdown('<h3 class="section-header">📊 Statistical Summary</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature ranges
        st.write("**Feature Ranges in Selected Data:**")
        for feature in selected_features:
            feature_min = analyzer.df[feature].min()
            feature_max = analyzer.df[feature].max()
            st.write(f"- **{feature.title()}**: {feature_min:.3f} to {feature_max:.3f}")
    
    with col2:
        # Hit song characteristics
        hit_data = analyzer.df[analyzer.df['is_hit'] == 1]
        if len(hit_data) > 0:
            st.write("**Hit Song Averages:**")
            for feature in selected_features[:3]:  # Show first 3 for brevity
                hit_avg = hit_data[feature].mean()
                st.write(f"- **{feature.title()}**: {hit_avg:.3f}")
def show_hit_song_profiler(analyzer):
    st.markdown('<h2 class="section-header">🎯 Hit Song Profiler</h2>', unsafe_allow_html=True)
    
    st.info("🔍 **Discover what makes hit songs successful** - Analyze the secret recipe for popular music!")
    
    # Main configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Feature selection
        available_features = ['danceability', 'energy', 'valence', 'acousticness', 
                            'speechiness', 'instrumentalness', 'liveness', 'tempo', 'loudness']
        selected_features = st.multiselect(
            "Select features to analyze:",
            available_features,
            default=['danceability', 'energy', 'valence', 'acousticness']
        )
    
    with col2:
        # Genre filter
        all_genres = ['All'] + analyzer.df['track_genre'].value_counts().head(15).index.tolist()
        selected_genre = st.selectbox("Filter by genre:", all_genres)
    
    with col3:
        # Visualization type
        viz_type = st.selectbox(
            "Comparison type:",
            ['radar', 'bar', 'violin'],
            format_func=lambda x: {'radar': 'Radar Chart', 'bar': 'Bar Chart', 'violin': 'Violin Plot'}[x]
        )
    
    if not selected_features:
        st.warning("Please select at least one feature to analyze.")
        return
    
    try:
        # Generate hit song analysis
        fig, insights, hit_stats, non_hit_stats, differences = analyzer.create_hit_song_profiler(
            selected_features=selected_features,
            genre_filter=selected_genre if selected_genre != 'All' else None,
            comparison_type=viz_type
        )
        
        # Display the main comparison chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display insights
        st.markdown("---")
        st.markdown('<h3 class="section-header">🔍 Hit Song Insights</h3>', unsafe_allow_html=True)
        
        col_insights1, col_insights2 = st.columns(2)
        
        with col_insights1:
            for insight in insights[:len(insights)//2]:
                if insight.strip():
                    st.write(insight)
        
        with col_insights2:
            for insight in insights[len(insights)//2:]:
                if insight.strip():
                    st.write(insight)
        
        # Feature Threshold Analysis
        st.markdown("---")
        st.markdown('<h3 class="section-header">📊 Feature Threshold Analysis</h3>', unsafe_allow_html=True)
        
        st.info("Explore how hit rate changes with different feature values")
        
        threshold_feature = st.selectbox(
            "Analyze hit rate for feature:",
            selected_features,
            key="threshold_feature"
        )
        
        if threshold_feature:
            threshold_fig, bin_analysis = analyzer.create_feature_threshold_analysis(
                feature=threshold_feature,
                genre_filter=selected_genre if selected_genre != 'All' else None
            )
            
            st.plotly_chart(threshold_fig, use_container_width=True)
            
            # Show optimal range
            optimal_bin = bin_analysis.loc[bin_analysis['hit_rate'].idxmax()]
            st.write(f"🎯 **Optimal Range for {threshold_feature.title()}:**")
            st.write(f"   • **Range**: {optimal_bin['feature_bin']}")
            st.write(f"   • **Average Value**: {optimal_bin[threshold_feature]:.3f}")
            st.write(f"   • **Hit Rate**: {optimal_bin['hit_rate']:.1f}%")
            st.write(f"   • **Songs in Range**: {optimal_bin['song_count']}")
        
        # Statistical Summary
        st.markdown("---")
        st.markdown('<h3 class="section-header">📈 Statistical Summary</h3>', unsafe_allow_html=True)
        
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            st.write("**Hit Songs Statistics:**")
            stats_data = []
            for feature in selected_features:
                stats_data.append({
                    'Feature': feature,
                    'Hit Avg': hit_stats[feature],
                    'Non-Hit Avg': non_hit_stats[feature],
                    'Difference %': differences[feature]
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df.style.format({
                'Hit Avg': '{:.3f}',
                'Non-Hit Avg': '{:.3f}', 
                'Difference %': '{:+.1f}%'
            }), use_container_width=True)
        
        with col_stats2:
            st.write("**Practical Recommendations:**")
            for feature in selected_features[:4]:
                diff = differences[feature]
                if diff > 10:
                    st.write(f"✅ **Increase {feature}** - Hits have {diff:.1f}% more")
                elif diff < -10:
                    st.write(f"✅ **Decrease {feature}** - Hits have {abs(diff):.1f}% less")
                else:
                    st.write(f"➡️ **{feature.title()}** - Similar between hits and non-hits")
            
    except Exception as e:
        st.error(f"Error generating hit song analysis: {str(e)}")
        st.info("Please try selecting different features or genres. Some genres may have very few hit songs.")

def show_hit_prediction(analyzer):
    st.markdown('<h2 class="section-header">🎼 Hit Prediction</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Coming Soon!** This section will allow you to:
    - Upload 30-second audio clips
    - Analyze audio features using machine learning
    - Get hit probability predictions
    - See feature comparisons with popular songs
    """)
    
    # Placeholder for future audio upload
    st.subheader("Feature Analysis Preview")
    
    # Let users explore what features make hits
    st.write("**Explore what makes a hit song:**")
    
    feature_to_analyze = st.selectbox("Select feature to analyze:", 
                                     ['danceability', 'energy', 'valence', 'tempo'])
    
    # Show hit vs non-hit comparison for selected feature
    hit_avg = analyzer.df[analyzer.df['is_hit'] == 1][feature_to_analyze].mean()
    non_hit_avg = analyzer.df[analyzer.df['is_hit'] == 0][feature_to_analyze].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Average {feature_to_analyze.title()} - Hits", f"{hit_avg:.3f}")
    with col2:
        st.metric(f"Average {feature_to_analyze.title()} - Non-Hits", f"{non_hit_avg:.3f}")
        
    # Show difference
    difference = hit_avg - non_hit_avg
    st.write(f"**Difference:** {difference:.3f} ({'Higher' if difference > 0 else 'Lower'} in hit songs)")

def show_feature_insights(analyzer):
    st.markdown('<h2 class="section-header">📊 Feature Insights</h2>', unsafe_allow_html=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    
    # Select features for correlation
    features_for_corr = ['danceability', 'energy', 'loudness', 'valence', 
                        'tempo', 'acousticness', 'instrumentalness', 'popularity']
    
    corr_matrix = analyzer.df[features_for_corr].corr()
    
    fig = px.imshow(corr_matrix, 
                   title="Audio Features Correlation Heatmap",
                   color_continuous_scale='RdYlGn',
                   aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot to show relationships
    st.subheader("Feature Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("X-axis feature:", features_for_corr, index=0)
    with col2:
        y_feature = st.selectbox("Y-axis feature:", features_for_corr, index=1)
    
    fig_scatter = px.scatter(analyzer.df, x=x_feature, y=y_feature, 
                            color='is_hit',
                            title=f"{x_feature.title()} vs {y_feature.title()}",
                            color_discrete_sequence=['#FF4B4B', '#1DB954'],
                            labels={'is_hit': 'Hit Song'})
    st.plotly_chart(fig_scatter, use_container_width=True)

def show_genre_analysis(analyzer):
    st.markdown('<h2 class="section-header">🎯 Advanced Genre Analysis</h2>', unsafe_allow_html=True)
    
    st.info("🔍 **Compare music features across different genres** - Discover unique genre characteristics!")
    
    # Filters Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Genre selection
        top_genres = analyzer.df['track_genre'].value_counts().head(20).index.tolist()
        selected_genres = st.multiselect(
            "Select genres to compare:",
            top_genres,
            default=top_genres[:4]
        )
    
    with col2:
        # Feature selection
        available_features = ['danceability', 'energy', 'valence', 'acousticness', 
                            'speechiness', 'instrumentalness', 'liveness', 'tempo']
        selected_features = st.multiselect(
            "Select features to compare:",
            available_features,
            default=['danceability', 'energy', 'valence', 'acousticness']
        )
    
    with col3:
        # Visualization type
        viz_type = st.selectbox(
            "Visualization type:",
            ['radar', 'heatmap', 'bar'],
            format_func=lambda x: {'radar': 'Radar Chart', 'heatmap': 'Heatmap', 'bar': 'Bar Chart'}[x]
        )
    
    if not selected_genres or not selected_features:
        st.warning("Please select at least one genre and one feature to analyze.")
        return
    
    # Generate the genre analysis
    try:
        fig, insights, genre_stats = analyzer.create_genre_comparison_dashboard(
            selected_genres=selected_genres,
            selected_features=selected_features,
            comparison_type=viz_type
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display insights
        st.markdown("---")
        st.markdown('<h3 class="section-header">🔍 Genre Insights</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            for insight in insights[:len(insights)//2]:
                if insight.strip():
                    st.write(insight)
        
        with col2:
            for insight in insights[len(insights)//2:]:
                if insight.strip():
                    st.write(insight)
        
        # Statistical Summary
        st.markdown("---")
        st.markdown('<h3 class="section-header">📊 Genre Statistics</h3>', unsafe_allow_html=True)
        
        # Display genre statistics table - SAFE VERSION
        display_columns = ['track_genre'] + selected_features
        if 'popularity_mean' in genre_stats.columns:
            display_columns.append('popularity_mean')
        if 'is_hit_mean' in genre_stats.columns:
            display_columns.append('is_hit_mean')
        
        # Only include columns that actually exist
        available_columns = [col for col in display_columns if col in genre_stats.columns]
        display_stats = genre_stats[available_columns].copy()
        
        # Format hit rate if available
        if 'is_hit_mean' in display_stats.columns:
            display_stats['hit_rate'] = (display_stats['is_hit_mean'] * 100).round(1)
            display_stats = display_stats.drop('is_hit_mean', axis=1)
        
        st.dataframe(display_stats, use_container_width=True)
        
        # Feature correlations by genre
        st.markdown("---")
        st.markdown('<h4 class="section-header">🔗 Feature Relationships by Genre</h4>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X-axis feature:", selected_features, key="genre_x")
        with col2:
            y_feature = st.selectbox("Y-axis feature:", selected_features, key="genre_y")
        
        if x_feature and y_feature:
            fig_scatter = px.scatter(analyzer.df[analyzer.df['track_genre'].isin(selected_genres)], 
                                   x=x_feature, y=y_feature, color='track_genre',
                                   title=f"{x_feature.title()} vs {y_feature.title()} by Genre",
                                   hover_data=['popularity'],
                                   opacity=0.7)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error generating genre analysis: {str(e)}")
        st.info("Please try selecting different genres or features.")
        
def show_feature_insights(analyzer):
    st.markdown('<h2 class="section-header">📊 Advanced Correlation Explorer</h2>', unsafe_allow_html=True)
    
    st.info("🔍 **Explore complex relationships between audio features** - Discover what drives music popularity!")
    
    # Main filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Feature selection
        available_features = ['danceability', 'energy', 'loudness', 'valence', 
                            'tempo', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'popularity']
        selected_features = st.multiselect(
            "Select features to analyze:",
            available_features,
            default=['danceability', 'energy', 'valence', 'acousticness', 'popularity']
        )
    
    with col2:
        # Color by option
        color_by = st.selectbox(
            "Color points by:",
            ['is_hit', 'track_genre', 'none'],
            format_func=lambda x: {'is_hit': 'Hit Status', 'track_genre': 'Genre', 'none': 'None'}[x]
        )
    
    with col3:
        # Genre filter
        all_genres = ['All'] + analyzer.df['track_genre'].value_counts().head(10).index.tolist()
        selected_genre = st.selectbox("Filter by genre:", all_genres)
    
    with col4:
        # Popularity range
        min_pop, max_pop = st.slider(
            "Popularity range:",
            min_value=0,
            max_value=100,
            value=(0, 100),
            key="correlation_popularity"
        )
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features to analyze correlations.")
        return
    
    # Generate correlation analysis
    try:
        scatter_matrix_fig, heatmap_fig, importance_fig, insights = analyzer.create_correlation_explorer(
            selected_features=selected_features,
            color_by=color_by,
            genre_filter=selected_genre if selected_genre != 'All' else None,
            popularity_range=(min_pop, max_pop)
        )
        
        # Display scatter plot matrix
        st.markdown("### 🔄 Scatter Plot Matrix")
        st.plotly_chart(scatter_matrix_fig, use_container_width=True)
        
        # Display correlation visualizations in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎨 Correlation Heatmap")
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Feature Importance")
            st.plotly_chart(importance_fig, use_container_width=True)
        
        # Display insights
        st.markdown("---")
        st.markdown('<h3 class="section-header">🔍 Correlation Insights</h3>', unsafe_allow_html=True)
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            for insight in insights[:len(insights)//2]:
                if insight.strip():
                    st.write(insight)
        
        with insights_col2:
            for insight in insights[len(insights)//2:]:
                if insight.strip():
                    st.write(insight)
        
        # 3D Visualization Section
        st.markdown("---")
        st.markdown('<h3 class="section-header">🎮 3D Feature Explorer</h3>', unsafe_allow_html=True)
        
        st.info("Explore your data in 3D space for deeper insights!")
        
        col_3d_1, col_3d_2, col_3d_3, col_3d_4 = st.columns(4)
        
        with col_3d_1:
            x_3d = st.selectbox("X-axis:", selected_features, index=0, key="3d_x")
        with col_3d_2:
            y_3d = st.selectbox("Y-axis:", selected_features, index=1, key="3d_y")
        with col_3d_3:
            z_3d = st.selectbox("Z-axis:", selected_features, index=2, key="3d_z")
        with col_3d_4:
            color_3d = st.selectbox("Color by:", ['is_hit', 'track_genre'], 
                                  format_func=lambda x: {'is_hit': 'Hit Status', 'track_genre': 'Genre'}[x],
                                  key="3d_color")
        
        if x_3d and y_3d and z_3d:
            fig_3d = analyzer.create_3d_correlation_plot(
                x_feature=x_3d,
                y_feature=y_3d,
                z_feature=z_3d,
                color_by=color_3d,
                genre_filter=selected_genre if selected_genre != 'All' else None
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error generating correlation analysis: {str(e)}")
        st.info("Please try selecting different features or adjusting filters.")
def show_hit_prediction(analyzer):
    st.markdown('<h2 class="section-header">🎯 Hit Song Profiler</h2>', unsafe_allow_html=True)
    
    st.info("🔍 **Discover what makes hit songs successful** - Analyze the secret recipe for popular music!")
    
    # Main configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Feature selection
        available_features = ['danceability', 'energy', 'valence', 'acousticness', 
                            'speechiness', 'instrumentalness', 'liveness', 'tempo', 'loudness']
        selected_features = st.multiselect(
            "Select features to analyze:",
            available_features,
            default=['danceability', 'energy', 'valence', 'acousticness']
        )
    
    with col2:
        # Genre filter
        all_genres = ['All'] + analyzer.df['track_genre'].value_counts().head(15).index.tolist()
        selected_genre = st.selectbox("Filter by genre:", all_genres)
    
    with col3:
        # Visualization type
        viz_type = st.selectbox(
            "Comparison type:",
            ['radar', 'bar', 'violin'],
            format_func=lambda x: {'radar': 'Radar Chart', 'bar': 'Bar Chart', 'violin': 'Violin Plot'}[x]
        )
    
    if not selected_features:
        st.warning("Please select at least one feature to analyze.")
        return
    
    try:
        # Generate hit song analysis
        fig, insights, hit_stats, non_hit_stats, differences = analyzer.create_hit_song_profiler(
            selected_features=selected_features,
            genre_filter=selected_genre if selected_genre != 'All' else None,
            comparison_type=viz_type
        )
        
        # Display the main comparison chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display insights
        st.markdown("---")
        st.markdown('<h3 class="section-header">🔍 Hit Song Insights</h3>', unsafe_allow_html=True)
        
        col_insights1, col_insights2 = st.columns(2)
        
        with col_insights1:
            for insight in insights[:len(insights)//2]:
                if insight.strip():
                    st.write(insight)
        
        with col_insights2:
            for insight in insights[len(insights)//2:]:
                if insight.strip():
                    st.write(insight)
        
        # Feature Threshold Analysis
        st.markdown("---")
        st.markdown('<h3 class="section-header">📊 Feature Threshold Analysis</h3>', unsafe_allow_html=True)
        
        st.info("Explore how hit rate changes with different feature values")
        
        threshold_feature = st.selectbox(
            "Analyze hit rate for feature:",
            selected_features,
            key="threshold_feature"
        )
        
        if threshold_feature:
            threshold_fig, bin_analysis = analyzer.create_feature_threshold_analysis(
                feature=threshold_feature,
                genre_filter=selected_genre if selected_genre != 'All' else None
            )
            
            st.plotly_chart(threshold_fig, use_container_width=True)
            
            # Show optimal range
            optimal_bin = bin_analysis.loc[bin_analysis['hit_rate'].idxmax()]
            st.write(f"🎯 **Optimal Range for {threshold_feature.title()}:**")
            st.write(f"   • **Range**: {optimal_bin['feature_bin']}")
            st.write(f"   • **Average Value**: {optimal_bin[threshold_feature]:.3f}")
            st.write(f"   • **Hit Rate**: {optimal_bin['hit_rate']:.1f}%")
            st.write(f"   • **Songs in Range**: {optimal_bin['song_count']}")
        
        # Statistical Summary
        st.markdown("---")
        st.markdown('<h3 class="section-header">📈 Statistical Summary</h3>', unsafe_allow_html=True)
        
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            st.write("**Hit Songs Statistics:**")
            stats_data = []
            for feature in selected_features:
                stats_data.append({
                    'Feature': feature,
                    'Hit Avg': hit_stats[feature],
                    'Non-Hit Avg': non_hit_stats[feature],
                    'Difference %': differences[feature]
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        
        with col_stats2:
            st.write("**Practical Recommendations:**")
            for feature in selected_features[:3]:
                diff = differences[feature]
                if diff > 10:
                    st.write(f"✅ **Increase {feature}** - Hits have {diff:.1f}% more")
                elif diff < -10:
                    st.write(f"✅ **Decrease {feature}** - Hits have {abs(diff):.1f}% less")
                else:
                    st.write(f"➡️ **{feature.title()}** - Similar between hits and non-hits")
            
    except Exception as e:
        st.error(f"Error generating hit song analysis: {str(e)}")
        st.info("Please try selecting different features or genres. Some genres may have very few hit songs.")

# Keep the old hit prediction as a separate page or integrate it
def show_ml_prediction(analyzer):
    """This will be the actual ML prediction page"""
    st.markdown('<h2 class="section-header">🤖 AI Hit Prediction</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Coming Soon!** This section will allow you to:
    - Upload 30-second audio clips
    - Analyze audio features using machine learning
    - Get hit probability predictions
    - See feature comparisons with popular songs
    """)
    
    # Placeholder for future audio upload
    st.subheader("Feature Analysis Preview")
    
    # Let users explore what features make hits
    st.write("**Explore what makes a hit song:**")
    
    feature_to_analyze = st.selectbox("Select feature to analyze:", 
                                     ['danceability', 'energy', 'valence', 'tempo'])
    
    # Show hit vs non-hit comparison for selected feature
    hit_avg = analyzer.df[analyzer.df['is_hit'] == 1][feature_to_analyze].mean()
    non_hit_avg = analyzer.df[analyzer.df['is_hit'] == 0][feature_to_analyze].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Average {feature_to_analyze.title()} - Hits", f"{hit_avg:.3f}")
    with col2:
        st.metric(f"Average {feature_to_analyze.title()} - Non-Hits", f"{non_hit_avg:.3f}")
        
    # Show difference
    difference = hit_avg - non_hit_avg
    st.write(f"**Difference:** {difference:.3f} ({'Higher' if difference > 0 else 'Lower'} in hit songs)")
def show_distribution_analyzer(analyzer):
    st.markdown('<h2 class="section-header">📈 Feature Distribution Analyzer</h2>', unsafe_allow_html=True)
    
    st.info("🔍 **Deep dive into feature distributions** - Understand the statistical properties of audio features!")
    
    # Main configuration
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Feature selection
        available_features = ['danceability', 'energy', 'valence', 'acousticness', 
                            'speechiness', 'instrumentalness', 'liveness', 'tempo', 'loudness']
        selected_features = st.multiselect(
            "Select features to analyze:",
            available_features,
            default=['danceability', 'energy', 'valence']
        )
    
    with col2:
        # Genre filter
        all_genres = ['All'] + analyzer.df['track_genre'].value_counts().head(15).index.tolist()
        selected_genre = st.selectbox("Filter by genre:", all_genres, key="dist_genre")
    
    with col3:
        # Distribution type
        dist_type = st.selectbox(
            "Distribution type:",
            ['histogram', 'density', 'box', 'qq', 'ecdf'],
            format_func=lambda x: {
                'histogram': 'Histogram', 
                'density': 'Density Plot', 
                'box': 'Box Plot',
                'qq': 'Q-Q Plot',
                'ecdf': 'Cumulative Distribution'
            }[x]
        )
    
    with col4:
        # Comparison option
        compare_hits = st.checkbox("Compare Hit vs Non-Hit", value=True)
    
    if not selected_features:
        st.warning("Please select at least one feature to analyze.")
        return
    
    try:
        # Generate distribution analysis
        fig, insights = analyzer.create_distribution_analyzer(
            selected_features=selected_features,
            genre_filter=selected_genre if selected_genre != 'All' else None,
            distribution_type=dist_type,
            compare_hits=compare_hits
        )
        
        # Display the distribution chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display insights
        st.markdown("---")
        st.markdown('<h3 class="section-header">🔍 Distribution Insights</h3>', unsafe_allow_html=True)
        
        col_insights1, col_insights2 = st.columns(2)
        
        with col_insights1:
            for insight in insights[:len(insights)//2]:
                if insight.strip():
                    st.write(insight)
        
        with col_insights2:
            for insight in insights[len(insights)//2:]:
                if insight.strip():
                    st.write(insight)
        
        # Statistical Summary Table
        st.markdown("---")
        st.markdown('<h3 class="section-header">📊 Statistical Summary</h3>', unsafe_allow_html=True)
        
        # Calculate comprehensive statistics
        stats_data = []
        filtered_df = analyzer.df.copy()
        if selected_genre != 'All':
            filtered_df = filtered_df[filtered_df['track_genre'] == selected_genre]
        
        for feature in selected_features:
            feature_data = filtered_df[feature].dropna()
            stats_data.append({
                'Feature': feature,
                'Count': len(feature_data),
                'Mean': feature_data.mean(),
                'Median': feature_data.median(),
                'Std Dev': feature_data.std(),
                'Min': feature_data.min(),
                'Max': feature_data.max(),
                'Skewness': feature_data.skew(),
                'Kurtosis': feature_data.kurtosis()
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df.round(4), use_container_width=True)
        
        # Outlier Analysis
        st.markdown("---")
        st.markdown('<h4 class="section-header">📏 Outlier Analysis</h4>', unsafe_allow_html=True)
        
        col_out1, col_out2 = st.columns(2)
        
        with col_out1:
            outlier_feature = st.selectbox("Analyze outliers for:", selected_features, key="outlier_feature")
        
        with col_out2:
            outlier_threshold = st.slider("Outlier threshold (z-score):", 1.0, 3.0, 2.0, 0.5)
        
        if outlier_feature:
            feature_data = filtered_df[outlier_feature].dropna()
            z_scores = np.abs((feature_data - feature_data.mean()) / feature_data.std())
            outliers = feature_data[z_scores > outlier_threshold]
            
            st.write(f"**Outliers in {outlier_feature}:** {len(outliers)} songs ({(len(outliers)/len(feature_data)*100):.1f}%)")
            
            if len(outliers) > 0:
                outlier_info = filtered_df[z_scores > outlier_threshold][[outlier_feature, 'track_name', 'artists', 'popularity']].head(10)
                st.dataframe(outlier_info, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error generating distribution analysis: {str(e)}")
        st.info("Please try selecting different features or adjusting filters.")
if __name__ == "__main__":
    main()