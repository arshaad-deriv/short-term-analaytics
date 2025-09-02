import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import glob

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌐 Translation Quality Intelligence: The Human Factor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .human-value-card {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .ai-limitation-card {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .critical-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Chart explanations dictionary
CHART_EXPLANATIONS = {
    "human_intervention": "Shows the percentage of AI/MT translations that required human post-editing. High percentages demonstrate that automated translation still needs significant human oversight.",
    
    "quality_improvement": "Compares approval rates between different translation methods. Higher human involvement typically correlates with better quality outcomes.",
    
    "error_severity": "Breaks down post-editing requirements by severity levels. Shows that humans catch and fix critical errors that could damage business reputation.",
    
    "temporal_reliability": "Tracks consistency of translation quality over time. Human-reviewed translations show more stable quality patterns.",
    
    "language_complexity": "Demonstrates how translation quality varies by language complexity. Shows where human expertise is most critical.",
    
    "provider_comparison": "Compares performance across different MT providers. Highlights the variability in automated translation quality.",
    
    "volume_quality": "Shows the relationship between translation volume and quality. Demonstrates scalability challenges of automated systems.",
    
    "cultural_adaptation": "Measures how well different methods handle cultural nuances and context-specific translations.",
    
    "business_risk": "Quantifies the business risk of relying solely on automated translation without human oversight.",
    
    "roi_analysis": "Calculates the return on investment of human translators in preventing costly translation errors."
}

def add_chart_explanation(title, explanation_key):
    """Add a help tooltip next to chart titles"""
    if explanation_key in CHART_EXPLANATIONS:
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.subheader(title)
        with col2:
            with st.expander("❓ Help"):
                st.write(CHART_EXPLANATIONS[explanation_key])
    else:
        st.subheader(title)

@st.cache_data
def load_crowdin_data():
    """Load and process all Crowdin JSON files"""
    json_files = glob.glob("*.json")
    
    if not json_files:
        st.error("No JSON files found in the current directory!")
        return None
    
    all_data = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            project_name = data.get('name', file_path.replace('.json', ''))
            date_range = data.get('dateRange', {})
            
            # Process each language's data
            for lang_data in data.get('data', []):
                language = lang_data.get('language', {})
                lang_name = language.get('name', 'Unknown')
                lang_code = language.get('code', 'unknown')
                
                # Process AI data
                ai_data = lang_data.get('ai', {})
                ai_cumulative = ai_data.get('cumulativeStatistics', {})
                ai_temporal = ai_data.get('temporalStatistics', {})
                
                if ai_cumulative and any(ai_cumulative.values()):
                    all_data.append({
                        'project': project_name,
                        'language': lang_name,
                        'language_code': lang_code,
                        'method': 'AI',
                        'approved_without_edit': ai_cumulative.get('approvedWithoutEdit', 0),
                        'post_edited_0_5': ai_cumulative.get('postEdited', {}).get('0-5', 0),
                        'post_edited_6_10': ai_cumulative.get('postEdited', {}).get('6-10', 0),
                        'post_edited_11_15': ai_cumulative.get('postEdited', {}).get('11-15', 0),
                        'post_edited_other': ai_cumulative.get('postEdited', {}).get('other', 0),
                        'weighted_units': ai_cumulative.get('weightedUnits', 0),
                        'temporal_data': ai_temporal,
                        'date_from': date_range.get('from'),
                        'date_to': date_range.get('to')
                    })
                
                # Process MT data
                mt_data = lang_data.get('mt', {})
                mt_cumulative = mt_data.get('cumulativeStatistics', {})
                mt_temporal = mt_data.get('temporalStatistics', {})
                
                if mt_cumulative and any(mt_cumulative.values()):
                    all_data.append({
                        'project': project_name,
                        'language': lang_name,
                        'language_code': lang_code,
                        'method': 'MT',
                        'approved_without_edit': mt_cumulative.get('approvedWithoutEdit', 0),
                        'post_edited_0_5': mt_cumulative.get('postEdited', {}).get('0-5', 0),
                        'post_edited_6_10': mt_cumulative.get('postEdited', {}).get('6-10', 0),
                        'post_edited_11_15': mt_cumulative.get('postEdited', {}).get('11-15', 0),
                        'post_edited_other': mt_cumulative.get('postEdited', {}).get('other', 0),
                        'weighted_units': mt_cumulative.get('weightedUnits', 0),
                        'temporal_data': mt_temporal,
                        'date_from': date_range.get('from'),
                        'date_to': date_range.get('to')
                    })
                
                # Process TM data
                tm_data = lang_data.get('tm', {})
                tm_cumulative = tm_data.get('cumulativeStatistics', {})
                tm_temporal = tm_data.get('temporalStatistics', {})
                
                if tm_cumulative and any(tm_cumulative.values()):
                    all_data.append({
                        'project': project_name,
                        'language': lang_name,
                        'language_code': lang_code,
                        'method': 'TM',
                        'approved_without_edit': tm_cumulative.get('approvedWithoutEdit', 0),
                        'post_edited_0_5': tm_cumulative.get('postEdited', {}).get('0-5', 0),
                        'post_edited_6_10': tm_cumulative.get('postEdited', {}).get('6-10', 0),
                        'post_edited_11_15': tm_cumulative.get('postEdited', {}).get('11-15', 0),
                        'post_edited_other': tm_cumulative.get('postEdited', {}).get('other', 0),
                        'weighted_units': tm_cumulative.get('weightedUnits', 0),
                        'temporal_data': tm_temporal,
                        'date_from': date_range.get('from'),
                        'date_to': date_range.get('to')
                    })
        
        except Exception as e:
            st.warning(f"Error processing {file_path}: {str(e)}")
            continue
    
    if not all_data:
        st.error("No valid data found in JSON files!")
        return None
    
    df = pd.DataFrame(all_data)
    
    # Calculate derived metrics
    df['total_strings'] = (df['approved_without_edit'] + df['post_edited_0_5'] + 
                          df['post_edited_6_10'] + df['post_edited_11_15'] + df['post_edited_other'])
    
    df['total_post_edited'] = (df['post_edited_0_5'] + df['post_edited_6_10'] + 
                              df['post_edited_11_15'] + df['post_edited_other'])
    
    # Calculate rates
    df['approval_rate'] = np.where(df['total_strings'] > 0, 
                                  (df['approved_without_edit'] / df['total_strings']) * 100, 0)
    
    df['human_intervention_rate'] = np.where(df['total_strings'] > 0,
                                           (df['total_post_edited'] / df['total_strings']) * 100, 0)
    
    df['critical_edit_rate'] = np.where(df['total_strings'] > 0,
                                       (df['post_edited_other'] / df['total_strings']) * 100, 0)
    
    df['minor_edit_rate'] = np.where(df['total_strings'] > 0,
                                    (df['post_edited_0_5'] / df['total_strings']) * 100, 0)
    
    # Quality score (weighted by edit severity)
    df['quality_score'] = np.where(df['total_strings'] > 0,
        (df['approved_without_edit'] * 100 + 
         df['post_edited_0_5'] * 95 + 
         df['post_edited_6_10'] * 85 + 
         df['post_edited_11_15'] * 70 + 
         df['post_edited_other'] * 40) / df['total_strings'], 0)
    
    # Risk score (higher = more risky)
    df['risk_score'] = (df['critical_edit_rate'] * 3 + 
                       df['post_edited_11_15'] / df['total_strings'] * 100 * 2 +
                       df['post_edited_6_10'] / df['total_strings'] * 100 * 1)
    
    return df

def create_temporal_data(df):
    """Extract and process temporal data"""
    temporal_records = []
    
    for _, row in df.iterrows():
        if row['temporal_data'] and isinstance(row['temporal_data'], dict):
            for date_str, stats in row['temporal_data'].items():
                try:
                    date_obj = pd.to_datetime(date_str)
                    total_day = (stats.get('approvedWithoutEdit', 0) + 
                               stats.get('postEdited', {}).get('0-5', 0) +
                               stats.get('postEdited', {}).get('6-10', 0) +
                               stats.get('postEdited', {}).get('11-15', 0) +
                               stats.get('postEdited', {}).get('other', 0))
                    
                    if total_day > 0:
                        approval_rate_day = (stats.get('approvedWithoutEdit', 0) / total_day) * 100
                        intervention_rate_day = ((stats.get('postEdited', {}).get('0-5', 0) +
                                                stats.get('postEdited', {}).get('6-10', 0) +
                                                stats.get('postEdited', {}).get('11-15', 0) +
                                                stats.get('postEdited', {}).get('other', 0)) / total_day) * 100
                        
                        temporal_records.append({
                            'date': date_obj,
                            'project': row['project'],
                            'language': row['language'],
                            'method': row['method'],
                            'approved_without_edit': stats.get('approvedWithoutEdit', 0),
                            'total_strings': total_day,
                            'approval_rate': approval_rate_day,
                            'intervention_rate': intervention_rate_day,
                            'critical_edits': stats.get('postEdited', {}).get('other', 0)
                        })
                except:
                    continue
    
    return pd.DataFrame(temporal_records) if temporal_records else pd.DataFrame()

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Translation Quality Intelligence: The Human Factor</h1>', unsafe_allow_html=True)
    # st.markdown("### Demonstrating the Irreplaceable Value of Human Translators in the AI Era")
    
    # Load data
    with st.spinner("🔄 Loading and analyzing Crowdin translation data..."):
        df = load_crowdin_data()
    
    if df is None or df.empty:
        st.error("❌ No data available. Please ensure JSON files are in the current directory.")
        return
    
    # Create temporal data
    temporal_df = create_temporal_data(df)
    
    # Sidebar filters
    st.sidebar.header("🎛️ Analysis Controls")
    
    # Project filter
    projects = st.sidebar.multiselect(
        "Select Projects",
        options=sorted(df['project'].unique()),
        default=sorted(df['project'].unique())[:5]  # Limit default selection
    )
    
    # Language filter
    languages = st.sidebar.multiselect(
        "Select Languages",
        options=sorted(df['language'].unique()),
        default=sorted(df['language'].unique())[:10]  # Limit default selection
    )
    
    # Method filter
    methods = st.sidebar.multiselect(
        "Select Methods",
        options=sorted(df['method'].unique()),
        default=sorted(df['method'].unique())
    )
    
    # Filter data
    filtered_df = df[
        (df['project'].isin(projects)) &
        (df['language'].isin(languages)) &
        (df['method'].isin(methods)) &
        (df['total_strings'] > 0)  # Only include records with actual data
    ]
    
    if filtered_df.empty:
        st.warning("⚠️ No data matches the selected filters. Please adjust your selection.")
        return
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Executive Summary", 
        "🧠 Human Value Proposition", 
        "⚠️ AI/MT Limitations", 
        "📊 Quality Analysis",
        "📈 Temporal Insights",
        "💼 Business Impact"
    ])
    
    with tab1:
        executive_summary(filtered_df, temporal_df)
    
    with tab2:
        human_value_proposition(filtered_df, temporal_df)
    
    with tab3:
        ai_mt_limitations(filtered_df, temporal_df)
    
    with tab4:
        quality_analysis(filtered_df, temporal_df)
    
    with tab5:
        temporal_insights(filtered_df, temporal_df)
    
    with tab6:
        business_impact(filtered_df, temporal_df)

def executive_summary(df, temporal_df):
    """Executive summary dashboard"""
    st.header("🎯 Executive Summary: The Translation Landscape")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_strings = df['total_strings'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_strings:,.0f}</h3>
            <p>Total Strings Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_intervention = df['human_intervention_rate'].mean()
        st.markdown(f"""
        <div class="human-value-card">
            <h3>{avg_intervention:.1f}%</h3>
            <p>Require Human Intervention</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        critical_errors = df['critical_edit_rate'].mean()
        st.markdown(f"""
        <div class="ai-limitation-card">
            <h3>{critical_errors:.1f}%</h3>
            <p>Critical Errors Caught</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        languages_count = df['language'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{languages_count}</h3>
            <p>Languages Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        projects_count = df['project'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{projects_count}</h3>
            <p>Projects Covered</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key insights
    st.subheader("🔍 Key Findings")
    
    # Calculate key insights
    high_intervention_methods = df[df['human_intervention_rate'] > 50]
    if not high_intervention_methods.empty:
        worst_method = high_intervention_methods.groupby('method')['human_intervention_rate'].mean().idxmax()
        worst_rate = high_intervention_methods.groupby('method')['human_intervention_rate'].mean().max()
        
        st.markdown(f"""
        <div class="critical-box" style='color: black;'>
            <strong>🚨 Critical Finding:</strong> {worst_method} translations require human intervention {worst_rate:.1f}% of the time, demonstrating that automated translation cannot operate independently.
        </div>
        """, unsafe_allow_html=True)
    
    # Quality comparison
    method_quality = df.groupby('method')['approval_rate'].mean().sort_values(ascending=False)
    if len(method_quality) > 1:
        best_method = method_quality.index[0]
        best_rate = method_quality.iloc[0]
        worst_method = method_quality.index[-1]
        worst_rate = method_quality.iloc[-1]
        
        st.markdown(f"""
        <div class="success-box" style='color: black;'>
            <strong>✅ Quality Gap:</strong> {best_method} achieves {best_rate:.1f}% approval rate vs {worst_method} at {worst_rate:.1f}%, showing significant variation in automated translation quality.
        </div>
        """, unsafe_allow_html=True)
    
    # Overview charts
    col1, col2 = st.columns(2)
    
    with col1:
        add_chart_explanation("Human Intervention Requirements by Method", "human_intervention")
        method_intervention = df.groupby('method')['human_intervention_rate'].mean().reset_index()
        fig = px.bar(
            method_intervention,
            x='method',
            y='human_intervention_rate',
            color='human_intervention_rate',
            color_continuous_scale='Reds',
            labels={'human_intervention_rate': 'Human Intervention Rate (%)'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        add_chart_explanation("Quality Distribution Across Methods", "quality_improvement")
        fig = px.box(
            df,
            x='method',
            y='approval_rate',
            color='method',
            labels={'approval_rate': 'Approval Rate (%)'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def human_value_proposition(df, temporal_df):
    """Human value proposition dashboard"""
    st.header("🧠 Human Value Proposition: Why Humans Are Irreplaceable")
    
    # Human intervention analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_intervention = df['human_intervention_rate'].mean()
        st.markdown(f"""
        <div class="human-value-card">
            <h3>{avg_intervention:.1f}%</h3>
            <p>Average Human Intervention Required</p>
            <small>Across all automated translations</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        critical_catch_rate = df[df['critical_edit_rate'] > 0]['critical_edit_rate'].mean()
        st.markdown(f"""
        <div class="human-value-card">
            <h3>{critical_catch_rate:.1f}%</h3>
            <p>Critical Errors Prevented</p>
            <small>By human reviewers</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        quality_improvement = df.groupby('method')['approval_rate'].mean()
        if 'TM' in quality_improvement.index and 'AI' in quality_improvement.index:
            improvement = quality_improvement['TM'] - quality_improvement['AI']
            st.markdown(f"""
            <div class="human-value-card">
                <h3>+{improvement:.1f}%</h3>
                <p>Quality Improvement</p>
                <small>TM vs AI (human-curated)</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed human value analysis
    st.subheader("📊 Human Intervention Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        add_chart_explanation("Post-Editing Severity Distribution", "error_severity")
        
        # Create post-editing breakdown
        edit_data = []
        for _, row in df.iterrows():
            if row['total_strings'] > 0:
                edit_data.extend([
                    {'Method': row['method'], 'Language': row['language'], 'Edit_Type': 'Minor (0-5%)', 'Count': row['post_edited_0_5']},
                    {'Method': row['method'], 'Language': row['language'], 'Edit_Type': 'Moderate (6-10%)', 'Count': row['post_edited_6_10']},
                    {'Method': row['method'], 'Language': row['language'], 'Edit_Type': 'Major (11-15%)', 'Count': row['post_edited_11_15']},
                    {'Method': row['method'], 'Language': row['language'], 'Edit_Type': 'Critical (>15%)', 'Count': row['post_edited_other']}
                ])
        
        edit_df = pd.DataFrame(edit_data)
        edit_summary = edit_df.groupby(['Method', 'Edit_Type'])['Count'].sum().reset_index()
        
        fig = px.bar(
            edit_summary,
            x='Method',
            y='Count',
            color='Edit_Type',
            color_discrete_map={
                'Minor (0-5%)': '#28a745',
                'Moderate (6-10%)': '#ffc107', 
                'Major (11-15%)': '#fd7e14',
                'Critical (>15%)': '#dc3545'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        add_chart_explanation("Language Complexity vs Human Intervention", "language_complexity")
        
        lang_analysis = df.groupby('language').agg({
            'human_intervention_rate': 'mean',
            'critical_edit_rate': 'mean',
            'total_strings': 'sum'
        }).reset_index()
        
        fig = px.scatter(
            lang_analysis,
            x='human_intervention_rate',
            y='critical_edit_rate',
            size='total_strings',
            hover_data=['language'],
            labels={
                'human_intervention_rate': 'Human Intervention Rate (%)',
                'critical_edit_rate': 'Critical Error Rate (%)'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Human value insights
    st.subheader("💡 Why Humans Remain Essential")
    
    # Calculate insights
    high_risk_languages = df[df['critical_edit_rate'] > df['critical_edit_rate'].quantile(0.75)]
    consistent_quality_methods = df.groupby('method')['approval_rate'].std().sort_values()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box" style='color: black;'>
            <h4>🎯 Quality Assurance</h4>
            <ul>
                <li>Humans catch critical errors that could damage brand reputation</li>
                <li>Provide consistent quality across different content types</li>
                <li>Ensure cultural appropriateness and context accuracy</li>
                <li>Maintain brand voice and tone consistency</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-box" style='color: black;'>
            <h4>🧠 Cognitive Advantages</h4>
            <ul>
                <li>Understanding of cultural nuances and idioms</li>
                <li>Context-aware decision making</li>
                <li>Creative problem-solving for complex translations</li>
                <li>Domain expertise in specialized fields</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def ai_mt_limitations(df, temporal_df):
    """AI/MT limitations dashboard"""
    st.header("⚠️ AI/MT Limitations: Where Automation Falls Short")
    
    # Limitation metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        failure_rate = df[df['approval_rate'] < 50].shape[0] / df.shape[0] * 100
        st.markdown(f"""
        <div class="ai-limitation-card">
            <h3>{failure_rate:.1f}%</h3>
            <p>Low Quality Combinations</p>
            <small>< 50% approval rate</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_intervention = df[df['human_intervention_rate'] > 70].shape[0] / df.shape[0] * 100
        st.markdown(f"""
        <div class="ai-limitation-card">
            <h3>{high_intervention:.1f}%</h3>
            <p>High Intervention Required</p>
            <small>> 70% need human help</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        inconsistent_quality = df.groupby('method')['approval_rate'].std().mean()
        st.markdown(f"""
        <div class="ai-limitation-card">
            <h3>{inconsistent_quality:.1f}</h3>
            <p>Quality Inconsistency</p>
            <small>Standard deviation</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        critical_errors = df['critical_edit_rate'].sum() / df['total_strings'].sum() * 100
        st.markdown(f"""
        <div class="ai-limitation-card">
            <h3>{critical_errors:.1f}%</h3>
            <p>Critical Error Rate</p>
            <small>Across all translations</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed limitation analysis
    col1, col2 = st.columns(2)
    
    with col1:
        add_chart_explanation("Quality Variability by Method", "temporal_reliability")
        
        # Quality consistency analysis
        method_stats = df.groupby('method').agg({
            'approval_rate': ['mean', 'std', 'min', 'max']
        }).round(2)
        method_stats.columns = ['Mean', 'Std Dev', 'Min', 'Max']
        method_stats = method_stats.reset_index()
        
        fig = go.Figure()
        
        for method in method_stats['method']:
            method_data = method_stats[method_stats['method'] == method].iloc[0]
            fig.add_trace(go.Box(
                y=[method_data['Min'], method_data['Mean'] - method_data['Std Dev'], 
                   method_data['Mean'], method_data['Mean'] + method_data['Std Dev'], method_data['Max']],
                name=method,
                boxmean=True
            ))
        
        fig.update_layout(
            title="Quality Range and Variability",
            yaxis_title="Approval Rate (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        add_chart_explanation("Failure Rate by Language Complexity", "language_complexity")
        
        # Language difficulty analysis
        lang_difficulty = df.groupby('language').agg({
            'approval_rate': 'mean',
            'human_intervention_rate': 'mean',
            'critical_edit_rate': 'mean',
            'total_strings': 'sum'
        }).reset_index()
        
        # Create difficulty score
        lang_difficulty['difficulty_score'] = (
            lang_difficulty['human_intervention_rate'] + 
            lang_difficulty['critical_edit_rate'] * 2
        )
        
        fig = px.scatter(
            lang_difficulty.nlargest(15, 'total_strings'),  # Top 15 by volume
            x='approval_rate',
            y='difficulty_score',
            size='total_strings',
            hover_data=['language'],
            labels={
                'approval_rate': 'Approval Rate (%)',
                'difficulty_score': 'Translation Difficulty Score'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk analysis
    st.subheader("🚨 Business Risk Analysis")
    
    # High-risk combinations
    high_risk = df[
        (df['critical_edit_rate'] > df['critical_edit_rate'].quantile(0.8)) |
        (df['approval_rate'] < 30)
    ].sort_values('risk_score', ascending=False)
    
    if not high_risk.empty:
        st.markdown("""
        <div class="critical-box" style='color: black;'>
            <h4>⚠️ High-Risk Translation Combinations</h4>
            <p>These combinations show concerning patterns that could lead to business risks:</p>
        </div>
        """, unsafe_allow_html=True)
        
        risk_display = high_risk[['project', 'language', 'method', 'approval_rate', 'critical_edit_rate', 'total_strings']].head(10)
        risk_display['approval_rate'] = risk_display['approval_rate'].round(1)
        risk_display['critical_edit_rate'] = risk_display['critical_edit_rate'].round(1)
        
        st.dataframe(risk_display, use_container_width=True)
    
    # Automation limitations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="warning-box" style='color: black;'>
            <h4>🤖 AI/MT Limitations</h4>
            <ul>
                <li><strong>Context Blindness:</strong> Cannot understand broader context</li>
                <li><strong>Cultural Insensitivity:</strong> Misses cultural nuances</li>
                <li><strong>Inconsistent Quality:</strong> Performance varies unpredictably</li>
                <li><strong>Domain Confusion:</strong> Struggles with specialized terminology</li>
                <li><strong>Creative Deficit:</strong> Cannot handle creative or marketing content</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="critical-box" style='color: black;'>
            <h4>💼 Business Consequences</h4>
            <ul>
                <li><strong>Brand Damage:</strong> Poor translations harm reputation</li>
                <li><strong>Customer Loss:</strong> Confusing content drives users away</li>
                <li><strong>Legal Risk:</strong> Mistranslations in legal/financial content</li>
                <li><strong>Market Failure:</strong> Cultural missteps in new markets</li>
                <li><strong>Compliance Issues:</strong> Regulatory translation errors</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def quality_analysis(df, temporal_df):
    """Quality analysis dashboard"""
    st.header("📊 Comprehensive Quality Analysis")
    
    # Quality metrics overview
    col1, col2 = st.columns(2)
    
    with col1:
        add_chart_explanation("Quality Score Distribution", "quality_improvement")
        
        fig = px.histogram(
            df,
            x='quality_score',
            color='method',
            nbins=30,
            labels={'quality_score': 'Quality Score', 'count': 'Frequency'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        add_chart_explanation("Method Performance Comparison", "provider_comparison")
        
        method_comparison = df.groupby('method').agg({
            'approval_rate': ['mean', 'std'],
            'human_intervention_rate': 'mean',
            'critical_edit_rate': 'mean',
            'total_strings': 'sum'
        }).round(2)
        
        method_comparison.columns = ['Approval_Mean', 'Approval_Std', 'Intervention_Rate', 'Critical_Rate', 'Total_Strings']
        method_comparison = method_comparison.reset_index()
        
        fig = px.bar(
            method_comparison,
            x='method',
            y='Approval_Mean',
            error_y='Approval_Std',
            color='method',
            labels={'Approval_Mean': 'Average Approval Rate (%)'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed quality breakdown
    st.subheader("🔍 Quality Deep Dive")
    
    # Project-level analysis
    add_chart_explanation("Project Quality Landscape", "volume_quality")
    
    project_quality = df.groupby('project').agg({
        'approval_rate': 'mean',
        'human_intervention_rate': 'mean',
        'total_strings': 'sum',
        'language': 'nunique'
    }).reset_index()
    
    fig = px.scatter(
        project_quality,
        x='total_strings',
        y='approval_rate',
        size='human_intervention_rate',
        color='language',
        hover_data=['project'],
        labels={
            'total_strings': 'Total Strings Processed',
            'approval_rate': 'Average Approval Rate (%)',
            'human_intervention_rate': 'Human Intervention Rate (%)',
            'language': 'Number of Languages'
        }
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Quality correlation analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Quality Correlations")
        
        # Calculate correlations
        correlations = df[['approval_rate', 'human_intervention_rate', 'critical_edit_rate', 'total_strings']].corr()
        
        fig = px.imshow(
            correlations,
            color_continuous_scale='RdBu',
            aspect="auto",
            labels={'color': 'Correlation Coefficient'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Quality Benchmarks")
        
        # Quality benchmarks
        benchmarks = {
            'Excellent (>90%)': len(df[df['approval_rate'] > 90]) / len(df) * 100,
            'Good (70-90%)': len(df[(df['approval_rate'] >= 70) & (df['approval_rate'] <= 90)]) / len(df) * 100,
            'Poor (50-70%)': len(df[(df['approval_rate'] >= 50) & (df['approval_rate'] < 70)]) / len(df) * 100,
            'Critical (<50%)': len(df[df['approval_rate'] < 50]) / len(df) * 100
        }
        
        benchmark_df = pd.DataFrame(list(benchmarks.items()), columns=['Quality_Level', 'Percentage'])
        
        fig = px.pie(
            benchmark_df,
            values='Percentage',
            names='Quality_Level',
            color_discrete_map={
                'Excellent (>90%)': '#28a745',
                'Good (70-90%)': '#17a2b8',
                'Poor (50-70%)': '#ffc107',
                'Critical (<50%)': '#dc3545'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def temporal_insights(df, temporal_df):
    """Temporal insights dashboard"""
    st.header("📈 Temporal Insights: Quality Trends Over Time")
    
    if temporal_df.empty:
        st.warning("⚠️ No temporal data available for analysis.")
        return
    
    # Temporal overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = (temporal_df['date'].max() - temporal_df['date'].min()).days
        st.markdown(f"""
        <div class="metric-card">
            <h3>{date_range}</h3>
            <p>Days of Data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_daily_volume = temporal_df.groupby('date')['total_strings'].sum().mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_daily_volume:.0f}</h3>
            <p>Avg Daily Volume</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        quality_trend = temporal_df.groupby('date')['approval_rate'].mean()
        if len(quality_trend) > 1:
            trend_slope = np.polyfit(range(len(quality_trend)), quality_trend.values, 1)[0]
            trend_direction = "📈 Improving" if trend_slope > 0 else "📉 Declining"
        else:
            trend_direction = "📊 Stable"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{trend_direction}</h3>
            <p>Quality Trend</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Temporal analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        add_chart_explanation("Quality Trends Over Time", "temporal_reliability")
        
        daily_quality = temporal_df.groupby(['date', 'method'])['approval_rate'].mean().reset_index()
        
        fig = px.line(
            daily_quality,
            x='date',
            y='approval_rate',
            color='method',
            markers=True,
            labels={'approval_rate': 'Approval Rate (%)', 'date': 'Date'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        add_chart_explanation("Volume vs Quality Correlation", "volume_quality")
        
        daily_summary = temporal_df.groupby('date').agg({
            'total_strings': 'sum',
            'approval_rate': 'mean'
        }).reset_index()
        
        fig = px.scatter(
            daily_summary,
            x='total_strings',
            y='approval_rate',
            trendline="ols",
            labels={'total_strings': 'Daily Volume', 'approval_rate': 'Average Approval Rate (%)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal patterns
    if len(temporal_df) > 30:  # Only if we have enough data
        st.subheader("📅 Seasonal Patterns")
        
        temporal_df['day_of_week'] = temporal_df['date'].dt.day_name()
        temporal_df['month'] = temporal_df['date'].dt.month_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            dow_analysis = temporal_df.groupby('day_of_week')['approval_rate'].mean().reset_index()
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_analysis['day_of_week'] = pd.Categorical(dow_analysis['day_of_week'], categories=day_order, ordered=True)
            dow_analysis = dow_analysis.sort_values('day_of_week')
            
            fig = px.bar(
                dow_analysis,
                x='day_of_week',
                y='approval_rate',
                labels={'approval_rate': 'Average Approval Rate (%)', 'day_of_week': 'Day of Week'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            monthly_analysis = temporal_df.groupby('month')['approval_rate'].mean().reset_index()
            
            fig = px.line(
                monthly_analysis,
                x='month',
                y='approval_rate',
                markers=True,
                labels={'approval_rate': 'Average Approval Rate (%)', 'month': 'Month'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def business_impact(df, temporal_df):
    """Business impact dashboard"""
    st.header("💼 Business Impact: The ROI of Human Translators")
    
    # ROI calculations
    total_strings = df['total_strings'].sum()
    total_critical_errors = df['post_edited_other'].sum()
    
    # Estimated costs (these would be customized based on actual business metrics)
    cost_per_critical_error = 100  # Estimated cost of a critical translation error
    cost_per_human_hour = 50  # Estimated cost of human translator per hour
    strings_per_hour = 200  # Estimated strings a human can process per hour
    
    # Calculate potential savings
    critical_error_cost = total_critical_errors * cost_per_critical_error
    human_review_cost = (df['total_post_edited'].sum() / strings_per_hour) * cost_per_human_hour
    
    # ROI metrics
    col1, col2,  = st.columns(2)
        
    with col1:
        roi_ratio = (critical_error_cost / human_review_cost) if human_review_cost > 0 else 0
        st.markdown(f"""
        <div class="human-value-card">
            <h3>{roi_ratio:.1f}x</h3>
            <p>ROI Multiplier</p>
            <small>Return on human investment</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_reduction = (1 - df['critical_edit_rate'].mean() / 100) * 100
        st.markdown(f"""
        <div class="human-value-card">
            <h3>{risk_reduction:.1f}%</h3>
            <p>Risk Reduction</p>
            <small>Through human oversight</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Business impact analysis
    st.subheader("📊 Strategic Business Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        add_chart_explanation("Cost-Benefit Analysis", "roi_analysis")
        
        # Create cost-benefit visualization
        scenarios = ['AI Only', 'AI + Human Review', 'Human Only']
        
        # Estimated costs and quality for each scenario
        scenario_data = {
            'Scenario': scenarios,
            'Quality_Score': [60, 85, 95],  # Estimated quality scores
            'Cost_Index': [1, 1.5, 3],     # Relative cost index
            'Risk_Level': [8, 3, 1]        # Risk level (1-10 scale)
        }
        
        scenario_df = pd.DataFrame(scenario_data)
        
        fig = px.scatter(
            scenario_df,
            x='Cost_Index',
            y='Quality_Score',
            size='Risk_Level',
            color='Scenario',
            labels={
                'Cost_Index': 'Relative Cost',
                'Quality_Score': 'Quality Score',
                'Risk_Level': 'Business Risk Level'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        add_chart_explanation("Risk Assessment Matrix", "business_risk")
        
        # Risk assessment by language and method
        risk_matrix = df.pivot_table(
            values='critical_edit_rate',
            index='language',
            columns='method',
            aggfunc='mean',
            fill_value=0
        )
        
        # Limit to top languages by volume for readability
        top_languages = df.groupby('language')['total_strings'].sum().nlargest(10).index
        risk_matrix_filtered = risk_matrix.loc[risk_matrix.index.isin(top_languages)]
        
        fig = px.imshow(
            risk_matrix_filtered,
            color_continuous_scale='Reds',
            aspect="auto",
            labels={'color': 'Critical Error Rate (%)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Strategic recommendations
    st.subheader("🎯 Strategic Recommendations")
    
    # Calculate key insights for recommendations
    high_risk_combinations = df[df['critical_edit_rate'] > df['critical_edit_rate'].quantile(0.75)]
    low_quality_methods = df.groupby('method')['approval_rate'].mean().sort_values()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box" style='color: black;'>
            <h4>✅ Immediate Actions</h4>
            <ol>
                <li><strong>Mandatory Human Review:</strong> Implement for all critical content</li>
                <li><strong>Quality Thresholds:</strong> Set minimum approval rates by language</li>
                <li><strong>Risk-Based Routing:</strong> Auto-route high-risk combinations to humans</li>
                <li><strong>Continuous Monitoring:</strong> Track quality metrics in real-time</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-box" style='color: black;'>
            <h4>🚀 Long-term Strategy</h4>
            <ol>
                <li><strong>Hybrid Workflow:</strong> AI for efficiency, humans for quality</li>
                <li><strong>Specialist Teams:</strong> Domain experts for complex content</li>
                <li><strong>Quality Training:</strong> Upskill translators on new technologies</li>
                <li><strong>Performance Incentives:</strong> Reward quality over speed</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Final business case
    st.subheader("💡 The Business Case for Human Translators")
    
    st.markdown(f"""
    <div class="insight-box" style='color: black;'>
        <h4>📈 Key Business Insights</h4>
        <p><strong>Our analysis of {total_strings:,} translated strings across {df['project'].nunique()} projects reveals:</strong></p>
        <ul>
            <li>🎯 <strong>{df['human_intervention_rate'].mean():.1f}% of automated translations require human intervention</strong></li>
            <li>⚠️ <strong>{total_critical_errors:,} critical errors were caught and corrected by human reviewers</strong></li>
            <li>💰 <strong>Estimated ${critical_error_cost:,.0f} in potential business costs prevented</strong></li>
            <li>🏆 <strong>{roi_ratio:.1f}x return on investment for human quality assurance</strong></li>
        </ul>
        <p><strong>Conclusion:</strong> Human translators are not just valuable—they're essential for maintaining quality, 
        preventing business risks, and ensuring customer satisfaction in our global markets.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 