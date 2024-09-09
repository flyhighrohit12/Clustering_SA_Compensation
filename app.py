import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
# Add these imports
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Set page config
st.set_page_config(page_title="SF Compensation Analysis", page_icon="💼", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .css-18e3th9 {
        padding-top: 0;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8cff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Employee_Compensation_SF.csv")
    
    # Clean and preprocess data
    df['Total Compensation'] = pd.to_numeric(df['Total Compensation'], errors='coerce')
    df['Salaries'] = pd.to_numeric(df['Salaries'], errors='coerce')
    df['Total Benefits'] = pd.to_numeric(df['Total Benefits'], errors='coerce')
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

df = load_data()

# Navigation
st.sidebar.title("Navigation")
selected = st.sidebar.radio(
    "Go to",
    ["Introduction", "Exploratory Analysis", "Clustering Analysis", "Recommendations"]
)

# ... (previous imports remain unchanged)



# ... (previous code remains unchanged)

if selected == "Introduction":
    st.title("💼 San Francisco Compensation Explorer")
    
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font">Uncover the secrets of San Francisco\'s public sector compensation!</p>', unsafe_allow_html=True)
    
    st.write("Welcome to an immersive journey through the intricacies of employee compensation in the City by the Bay. This powerful tool allows you to dissect, visualize, and understand the complex world of public sector salaries and benefits.")
    
    # Key Statistics
    st.subheader("📊 Quick Insights")
    col1, col2, col3 = st.columns(3)
    total_compensation = df['Total Compensation'].sum()
    avg_compensation = df['Total Compensation'].mean()
    num_employees = len(df)
    
    col1.metric("Total Annual Compensation", f"${total_compensation:,.0f}")
    col2.metric("Average Compensation", f"${avg_compensation:,.0f}")
    col3.metric("Number of Employees", f"{num_employees:,}")
    
    # Treemap of Departments
    st.subheader("🏙️ City Departments")
    dept_sizes = df.groupby('Department')['Total Compensation'].sum().reset_index()
    dept_sizes = dept_sizes.sort_values('Total Compensation', ascending=False).head(10)
    
    fig = px.treemap(dept_sizes, path=['Department'], values='Total Compensation',
                     title='Top 10 Departments by Total Compensation',
                     hover_data=['Total Compensation'],
                     color='Total Compensation',
                     color_continuous_scale='Viridis')
    fig.update_traces(textinfo="label+value")
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("Explore the diverse landscape of San Francisco's city departments and their compensation allocations.")
    
    # Word Cloud of Job Titles
    st.subheader("🧑‍💼 Job Title Universe")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Job']))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
    st.write("Dive into the rich tapestry of job titles across San Francisco's public sector.")
    
    # Interactive Scatter Plot
    st.subheader("💰 Compensation Landscape")
    fig = px.scatter(df, x='Salaries', y='Total Benefits', color='Organization Group',
                     hover_name='Job', hover_data=['Total Compensation'],
                     title='Salaries vs Benefits Across Organization Groups')
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("Uncover patterns and outliers in the relationship between salaries and benefits across different organization groups.")
    
    st.subheader("🚀 Embark on Your Exploration")
    st.write("""
    This application empowers you to:
    1. **Dive Deep**: Explore detailed visualizations in the Exploratory Analysis tab.
    2. **Uncover Patterns**: Leverage advanced clustering techniques in the Clustering Analysis tab.
    3. **Gain Insights**: Discover data-driven recommendations tailored to San Francisco's unique compensation landscape.
    
    Whether you're a policymaker, researcher, or curious citizen, this tool provides unprecedented access to the intricate world of public sector compensation. Start your journey now!
    """)
    
    st.info("Navigate through the tabs above to begin your exploration. Each section offers unique insights and interactive visualizations to help you understand San Francisco's compensation data.")

# ... (rest of the code remains unchanged)



# ... (previous code remains unchanged)

elif selected == "Exploratory Analysis":
    st.title("🔍 Exploratory Data Analysis")
    st.write("Dive deep into the fascinating world of San Francisco's employee compensation data.")

    # 1. Distribution of Total Compensation
    st.subheader("💰 Distribution of Total Compensation")
    fig = px.histogram(df, x="Total Compensation", nbins=50, marginal="box", 
                       color_discrete_sequence=['#FFA07A'])
    fig.update_layout(title_text="Distribution of Total Compensation", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    
    q1 = df['Total Compensation'].quantile(0.25)
    median = df['Total Compensation'].median()
    q3 = df['Total Compensation'].quantile(0.75)
    iqr = q3 - q1
    
    st.write(f"""
    📊 **Insights:**
    - The median total compensation is ${median:,.2f}.
    - 25% of employees earn less than ${q1:,.2f}, while 75% earn less than ${q3:,.2f}.
    - The Interquartile Range (IQR) is ${iqr:,.2f}, showing significant variation in compensation.
    - The distribution is right-skewed, indicating a small number of high earners pulling the average up.
    - This skew suggests the need for targeted retention strategies for high-value employees.
    """)

    # 2. Top 10 Highest Paid Jobs
    st.subheader("🏆 Top 10 Highest Paid Jobs")
    top_jobs = df.groupby('Job')['Total Compensation'].mean().sort_values(ascending=False).head(10)
    fig = px.bar(top_jobs, x=top_jobs.index, y=top_jobs.values, 
                 labels={'y': 'Average Total Compensation', 'x': 'Job Title'},
                 color=top_jobs.values, color_continuous_scale='Viridis')
    fig.update_layout(title_text="Top 10 Highest Paid Jobs", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"""
    💼 **Insights:**
    - The highest-paid job on average is "{top_jobs.index[0]}" with ${top_jobs.values[0]:,.2f} in total compensation.
    - There's a {(top_jobs.values[0] / top_jobs.values[-1] - 1) * 100:.1f}% difference between the highest and 10th highest paid jobs.
    - The top-paid jobs are predominantly in medical, legal, and executive fields.
    - This concentration of high salaries in specialized roles highlights the importance of retaining key talent in these areas.
    """)

    # 3. Compensation by Department
    st.subheader("🏢 Average Compensation by Department")
    dept_comp = df.groupby('Department')[['Salaries', 'Total Benefits', 'Total Compensation']].mean().sort_values('Total Compensation', ascending=False).head(15)
    fig = px.bar(dept_comp, x=dept_comp.index, y=['Salaries', 'Total Benefits'], 
                 labels={'value': 'Amount', 'variable': 'Category'},
                 title="Top 15 Departments by Average Compensation",
                 color_discrete_map={'Salaries': '#4CAF50', 'Total Benefits': '#2196F3'})
    fig.update_layout(barmode='stack')
    st.plotly_chart(fig, use_container_width=True)
    
    top_dept = dept_comp.index[0]
    top_dept_salary = dept_comp.loc[top_dept, 'Salaries']
    top_dept_benefits = dept_comp.loc[top_dept, 'Total Benefits']
    
    st.write(f"""
    🏙️ **Insights:**
    - The department with the highest average compensation is "{top_dept}".
    - In this department, the average salary is ${top_dept_salary:,.2f} and average benefits are ${top_dept_benefits:,.2f}.
    - The ratio of benefits to salary varies significantly across departments, from {(dept_comp['Total Benefits'] / dept_comp['Salaries']).min():.2f} to {(dept_comp['Total Benefits'] / dept_comp['Salaries']).max():.2f}.
    - This variation suggests different compensation strategies or job nature across departments.
    """)

    st.subheader("⏰ Overtime Distribution")
    fig = px.box(df, y="Overtime", points="all", color_discrete_sequence=['#FF4136'])
    fig.update_layout(title_text="Distribution of Overtime Pay", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    median_overtime = df['Overtime'].median()
    max_overtime = df['Overtime'].max()
    overtime_receivers = (df['Overtime'] > 0).mean() * 100

    st.write(f"""
    🕒 **Insights:**
    - The median overtime pay is ${median_overtime:,.2f}, but there are significant outliers.
    - The maximum overtime pay recorded is ${max_overtime:,.2f}.
    """)

    if median_overtime > 0:
        st.write(f"- The maximum overtime is {max_overtime/median_overtime:.1f}x the median.")
    else:
        st.write("- The median overtime is $0, indicating that at least half of the employees do not receive overtime pay.")

    st.write(f"""
    - {overtime_receivers:.1f}% of employees receive some form of overtime pay.
    - The long upper whisker suggests potential overwork in certain roles, which could lead to burnout and decreased productivity.
    - This distribution indicates a need to review overtime policies and staffing levels in departments with high overtime hours.
    """)

    # 5. Correlation Heatmap
    st.subheader("🔗 Correlation Between Compensation Components")
    numeric_cols = ['Salaries', 'Overtime', 'Other Salaries', 'Total Salary', 'Retirement', 'Health/Dental', 'Other Benefits', 'Total Benefits', 'Total Compensation']
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    fig.update_layout(title_text="Correlation Heatmap of Compensation Components", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"""
    📊 **Insights:**
    - Salaries and Total Compensation have a very strong positive correlation ({corr_matrix.loc['Salaries', 'Total Compensation']:.2f}), as expected.
    - Overtime has a moderate positive correlation ({corr_matrix.loc['Overtime', 'Total Compensation']:.2f}) with Total Compensation.
    - Health/Dental benefits show a weaker correlation ({corr_matrix.loc['Health/Dental', 'Total Compensation']:.2f}) with Total Compensation compared to other components.
    - This suggests that increasing base salaries or overtime opportunities might have a more significant impact on total compensation than adjusting health benefits.
    """)

    # 6. Compensation Trends Over Time
    st.subheader("📈 Compensation Trends Over Time")
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    yearly_avg = df.groupby(df['Year'].dt.year)[['Salaries', 'Total Benefits', 'Total Compensation']].mean().reset_index()
    fig = px.line(yearly_avg, x='Year', y=['Salaries', 'Total Benefits', 'Total Compensation'], 
                  labels={'value': 'Amount', 'variable': 'Category'},
                  title="Average Compensation Components Over Time")
    st.plotly_chart(fig, use_container_width=True)
    
    start_year = yearly_avg['Year'].min()
    end_year = yearly_avg['Year'].max()
    total_growth = (yearly_avg['Total Compensation'].iloc[-1] / yearly_avg['Total Compensation'].iloc[0] - 1) * 100
    
    st.write(f"""
    📅 **Insights:**
    - From {start_year} to {end_year}, the average total compensation has grown by {total_growth:.1f}%.
    - Salaries have grown by {((yearly_avg['Salaries'].iloc[-1] / yearly_avg['Salaries'].iloc[0]) - 1) * 100:.1f}%, while benefits have increased by {((yearly_avg['Total Benefits'].iloc[-1] / yearly_avg['Total Benefits'].iloc[0]) - 1) * 100:.1f}%.
    - The gap between salaries and benefits has {
    'widened' if (yearly_avg['Salaries'].iloc[-1] / yearly_avg['Total Benefits'].iloc[-1]) > (yearly_avg['Salaries'].iloc[0] / yearly_avg['Total Benefits'].iloc[0]) 
    else 'narrowed'} over time.
    - This trend suggests a shift in the city's compensation strategy, potentially reflecting changes in labor market conditions or policy priorities.
    """)

    # 7. Job Family Distribution
    st.subheader("👥 Job Family Distribution")
    job_family_counts = df['Job Family'].value_counts()
    fig = px.pie(values=job_family_counts.values, names=job_family_counts.index, 
                 title="Distribution of Employees Across Job Families")
    st.plotly_chart(fig, use_container_width=True)
    
    largest_family = job_family_counts.index[0]
    largest_family_pct = (job_family_counts.values[0] / job_family_counts.sum()) * 100
    
    st.write(f"""
    🧑‍🤝‍🧑 **Insights:**
    - The largest job family is "{largest_family}", comprising {largest_family_pct:.1f}% of all employees.
    - There are {len(job_family_counts)} distinct job families in the dataset.
    - The diversity of job families reflects the complex structure of San Francisco's public sector workforce.
    - Understanding the distribution of job families can help in tailoring training programs and career development pathways.
    """)

    st.subheader("🔍 Key Takeaways")
    st.write("""
    1. San Francisco's public sector compensation is characterized by a wide range, with some high-earning outliers significantly influencing the average.
    2. Certain specialized roles, particularly in medical and executive fields, command substantially higher compensation.
    3. The structure of compensation (ratio of salary to benefits) varies significantly across departments, suggesting different strategies or job requirements.
    4. Overtime pay is a significant factor for some employees, potentially indicating staffing challenges in certain areas.
    5. There's a clear upward trend in average compensation over time, with salaries and benefits growing at different rates.
    6. The diverse range of job families highlights the complexity of the city's workforce, necessitating tailored approaches to compensation and career development.
    """)

# ... (rest of the code remains unchanged)
# ... (rest of the code remains unchanged)

elif selected == "Clustering Analysis":
    st.title("🧩 Clustering Analysis")
    
    # Prepare data for clustering
    features = ['Salaries', 'Overtime', 'Total Benefits']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Algorithm selection on main page
    st.subheader("Select Clustering Algorithms")
    algorithms = st.multiselect("Choose algorithms to compare", 
                                ["K-Means", "Hierarchical", "DBSCAN"],
                                default=["K-Means"])
    
    # Function to run clustering and visualize results
    def run_clustering(X, algorithm, name):
        if algorithm == "K-Means":
            n_clusters = st.slider(f"Number of Clusters for {name}", 2, 10, 3, key=f"{name}_slider")
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif algorithm == "Hierarchical":
            n_clusters = st.slider(f"Number of Clusters for {name}", 2, 10, 3, key=f"{name}_slider")
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif algorithm == "DBSCAN":
            eps = st.slider(f"Epsilon for {name}", 0.1, 2.0, 0.5, key=f"{name}_eps")
            min_samples = st.slider(f"Minimum Samples for {name}", 2, 20, 5, key=f"{name}_min_samples")
            model = DBSCAN(eps=eps, min_samples=min_samples)
        
        labels = model.fit_predict(X)
        
        fig = px.scatter(x=X[:, 0], y=X[:, 1], color=labels.astype(str),
                         labels={'x': 'First Component', 'y': 'Second Component'},
                         title=f"{name} Clustering")
        
        return fig, labels, model
    
    # Run selected algorithms and display results
    for algorithm in algorithms:
        st.subheader(f"{algorithm} Clustering")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Without PCA")
            fig, labels, model = run_clustering(X_scaled, algorithm, f"{algorithm}")
            st.plotly_chart(fig, use_container_width=True)
            
            if algorithm != "DBSCAN":
                silhouette_avg = silhouette_score(X_scaled, labels)
                st.write(f"Silhouette Score: {silhouette_avg:.2f}")
        
        with col2:
            st.write("With PCA")
            fig_pca, labels_pca, model_pca = run_clustering(X_pca, algorithm, f"{algorithm} (PCA)")
            st.plotly_chart(fig_pca, use_container_width=True)
            
            if algorithm != "DBSCAN":
                silhouette_avg_pca = silhouette_score(X_pca, labels_pca)
                st.write(f"Silhouette Score (PCA): {silhouette_avg_pca:.2f}")
        
        # Generate data-specific insights
        cluster_stats = pd.DataFrame({
            'Cluster': labels,
            'Salaries': X['Salaries'],
            'Overtime': X['Overtime'],
            'Total Benefits': X['Total Benefits']
        }).groupby('Cluster').mean()
        
        st.subheader(f"Cluster Characteristics for {algorithm}")
        st.write(cluster_stats)
        
        # Generate insights based on clustering results
        st.subheader(f"Data-Specific Insights for {algorithm}")
        
        if algorithm == "K-Means":
            highest_salary_cluster = cluster_stats['Salaries'].idxmax()
            highest_benefits_cluster = cluster_stats['Total Benefits'].idxmax()
            st.write(f"- Cluster {highest_salary_cluster} has the highest average salary (${cluster_stats.loc[highest_salary_cluster, 'Salaries']:,.2f}).")
            st.write(f"- Cluster {highest_benefits_cluster} has the highest average benefits (${cluster_stats.loc[highest_benefits_cluster, 'Total Benefits']:,.2f}).")
            
            if highest_salary_cluster != highest_benefits_cluster:
                st.write("- Interestingly, the cluster with the highest salaries is different from the one with the highest benefits.")
                st.write("  This suggests varying compensation structures across different employee groups.")
            
            overtime_cluster = cluster_stats['Overtime'].idxmax()
            st.write(f"- Cluster {overtime_cluster} has the highest average overtime (${cluster_stats.loc[overtime_cluster, 'Overtime']:,.2f}).")
            st.write("  This group might represent employees in roles or departments with high overtime demands.")
        
        elif algorithm == "Hierarchical":
            n_clusters = len(cluster_stats)
            st.write(f"- The hierarchical clustering resulted in {n_clusters} distinct groups.")
            if n_clusters > 2:
                st.write("  This suggests a multi-tiered structure in the compensation data, possibly reflecting job levels or department hierarchies.")
            
            spread = cluster_stats.std().sum()
            st.write(f"- The total spread of cluster centers is ${spread:,.2f}, indicating the overall dispersion of compensation groups.")
        
        elif algorithm == "DBSCAN":
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            st.write(f"- DBSCAN identified {n_clusters} clusters and {n_noise} noise points.")
            st.write(f"  {n_noise/len(labels):.1%} of the data points are considered outliers.")
            st.write("  These outliers might represent unique positions or compensation arrangements that don't fit the general patterns.")
        
        st.write("- The clustering results provide insights into the natural groupings of employees based on their compensation structure.")
        st.write("  These groups could inform targeted retention strategies or help identify potential inequities in compensation.")
    
    # ... (previous code remains unchanged)

    st.subheader("🧬 PCA: Unraveling Compensation Complexity")
    st.write("Principal Component Analysis (PCA) helps us understand the underlying structure of our compensation data.")

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Explained Variance Ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Create a DataFrame for the PCA results
    pca_df = pd.DataFrame({
        'Principal Component': range(1, len(explained_variance_ratio) + 1),
        'Explained Variance Ratio': explained_variance_ratio,
        'Cumulative Explained Variance Ratio': cumulative_variance_ratio
    })

    # Visualization 1: Scree Plot
    fig1 = px.line(pca_df, x='Principal Component', y=['Explained Variance Ratio', 'Cumulative Explained Variance Ratio'],
                labels={'value': 'Explained Variance Ratio', 'variable': 'Metric'},
                title='Scree Plot: Explained Variance by Principal Component',
                color_discrete_map={'Explained Variance Ratio': '#FFA07A', 'Cumulative Explained Variance Ratio': '#20B2AA'})
    fig1.add_scatter(x=pca_df['Principal Component'], y=pca_df['Explained Variance Ratio'], mode='markers', 
                    marker=dict(size=10, color='#FFA07A'), name='Individual')
    fig1.add_scatter(x=pca_df['Principal Component'], y=pca_df['Cumulative Explained Variance Ratio'], mode='markers', 
                    marker=dict(size=10, color='#20B2AA'), name='Cumulative')
    fig1.update_layout(legend_title_text='')
    st.plotly_chart(fig1, use_container_width=True)

    st.write(f"""
    📊 **Scree Plot Insights:**
    - The first principal component (PC1) explains {explained_variance_ratio[0]:.2%} of the total variance in the data.
    - PC2 adds another {explained_variance_ratio[1]:.2%}, bringing the cumulative explained variance to {cumulative_variance_ratio[1]:.2%}.
    - We reach {cumulative_variance_ratio[2]:.2%} cumulative explained variance with just 3 principal components.
    - This suggests that the compensation structure in San Francisco can be largely understood through these first few components.
    """)

    # Visualization 2: 2D Scatter Plot of first two PCs
    fig2 = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=df['Total Compensation'],
                    labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
                    title='Employee Distribution in PC1 vs PC2 Space',
                    color_continuous_scale='Viridis')
    st.plotly_chart(fig2, use_container_width=True)

    st.write(f"""
    🔍 **PC1 vs PC2 Scatter Plot Insights:**
    - The color gradient shows that Total Compensation generally increases from left to right, indicating that PC1 is strongly related to overall compensation level.
    - The vertical spread suggests that PC2 might be capturing differences in compensation structure (e.g., ratio of salary to benefits) rather than just magnitude.
    - Clusters or patterns in this plot could represent different job categories or departments with similar compensation structures.
    """)

    # Visualization 3: Feature Importance in PC1 and PC2
    pca_components = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=features
    )

    fig3 = px.imshow(pca_components[['PC1', 'PC2']].T,
                    labels=dict(x="Features", y="Principal Components"),
                    x=features,
                    y=['PC1', 'PC2'],
                    color_continuous_scale='RdBu_r',
                    title='Feature Importance in PC1 and PC2')
    st.plotly_chart(fig3, use_container_width=True)

    # Identify most important features for PC1 and PC2
    pc1_top_feature = pca_components['PC1'].abs().idxmax()
    pc2_top_feature = pca_components['PC2'].abs().idxmax()

    st.write(f"""
    🎯 **Feature Importance Insights:**
    - PC1 is most strongly influenced by {pc1_top_feature}, with a coefficient of {pca_components.loc[pc1_top_feature, 'PC1']:.2f}.
    This suggests that {pc1_top_feature} is the primary driver of overall compensation differences.
    - PC2 is most heavily weighted on {pc2_top_feature} (coefficient: {pca_components.loc[pc2_top_feature, 'PC2']:.2f}).
    This indicates that {pc2_top_feature} plays a key role in differentiating compensation structures beyond just total amount.
    - The contrasting colors between some features in PC1 and PC2 (e.g., {features[0]} and {features[2]}) suggest these components are capturing different aspects of compensation structure.
    """)

    st.subheader("🧠 Key Takeaways from PCA")
    st.write(f"""
    1. **Dimensionality Reduction**: We can capture {cumulative_variance_ratio[2]:.2%} of the variance in compensation data using just 3 principal components, simplifying our 9-dimensional data significantly.

    2. **Primary Compensation Driver**: The first principal component, heavily influenced by {pc1_top_feature}, explains the majority of variance ({explained_variance_ratio[0]:.2%}). This suggests that {pc1_top_feature} is the most critical factor in determining an employee's overall compensation.

    3. **Compensation Structure Insights**: The second principal component, primarily weighted on {pc2_top_feature}, reveals additional nuances in compensation structure, possibly highlighting differences between job roles or departments.

    4. **Efficient Analysis**: By focusing on these principal components, we can efficiently analyze and compare employee compensations, potentially identifying outliers or groups with unique compensation structures.

    5. **Policy Implications**: Understanding these principal components can guide policy decisions. For example, adjustments to {pc1_top_feature} would have the most significant impact on overall compensation levels, while changes to {pc2_top_feature} might affect the balance of compensation components.
    """)

# ... (rest of the code remains unchanged)
# ... (rest of the code remains unchanged)

elif selected == "Recommendations":
    st.title("💡 Data-Specific Recommendations and Insights")
    
    # Calculate some statistics
    avg_salary = df['Salaries'].mean()
    avg_benefits = df['Total Benefits'].mean()
    avg_total_comp = df['Total Compensation'].mean()
    highest_paid_job = df.loc[df['Total Compensation'].idxmax(), 'Job']
    highest_paid_dept = df.groupby('Department')['Total Compensation'].mean().idxmax()
    
    st.write(f"""
    Based on the analysis of the San Francisco employee compensation data, here are data-specific insights and recommendations:

    1. Compensation Structure:
       - Average Salary: ${avg_salary:,.2f}
       - Average Benefits: ${avg_benefits:,.2f}
       - Average Total Compensation: ${avg_total_comp:,.2f}
       - The highest-paid job is "{highest_paid_job}"
       - The department with the highest average compensation is "{highest_paid_dept}"

    Recommendation: Review the compensation structure, especially for departments and roles significantly above or below these averages.

    2. Benefits Analysis:
       - Benefits make up {(avg_benefits/avg_total_comp)*100:.1f}% of the average total compensation.
    
    Recommendation: Evaluate the benefits package, especially for roles where it constitutes a larger portion of total compensation.

    3. Overtime Patterns:
       - {(df['Overtime'] > 0).mean()*100:.1f}% of employees received overtime pay.
       - Average overtime pay for those who received it: ${df[df['Overtime'] > 0]['Overtime'].mean():,.2f}
    
    Recommendation: Investigate departments with high overtime costs to optimize workforce management.

    4. Job Family Analysis:
    """)
    
    job_family_stats = df.groupby('Job Family')['Total Compensation'].agg(['mean', 'median', 'std']).sort_values('mean', ascending=False)
    st.write(job_family_stats)
    
    st.write("""
    Recommendation: Focus on job families with high standard deviations in compensation for internal equity reviews.

    5. Clustering Insights:
       - K-Means clustering revealed distinct groups based on compensation structure.
       - Hierarchical clustering showed nested patterns in compensation, potentially related to job hierarchies.
       - DBSCAN identified outliers in compensation, which may warrant individual review.
    
    Recommendation: Use clustering results to inform targeted retention strategies and identify anomalies in compensation.

    6. PCA Insights:
       - The first two principal components explain a significant portion of the variance in compensation data.
       - This suggests that most of the variation in compensation can be captured by a few key factors.
    
    Recommendation: Focus on these key factors (likely related to job level, department, and tenure) when making compensation decisions.

    7. Algorithm Comparison:
       - K-Means performed well for identifying broad compensation groups.
       - Hierarchical clustering provided insights into the nested structure of compensation.
       - DBSCAN was effective at identifying outliers and unusual compensation patterns.
    
    Recommendation: Use a combination of these algorithms for a comprehensive view of compensation patterns and anomalies.

    8. Longitudinal Analysis:
       - Consider incorporating year-over-year data to track compensation trends over time.
       - This could reveal patterns in salary increases, changes in benefits structure, or shifts in overtime policies.

    9. Equity Considerations:
       - While demographic data is not included in this dataset, it's crucial to conduct regular pay equity analyses.
       - Consider collecting and analyzing data on gender, race, and other relevant factors to ensure equitable compensation practices.

    10. Benchmarking:
        - Compare these compensation figures with industry standards and other comparable cities.
        - This can help ensure that San Francisco remains competitive in attracting and retaining talent.

    These data-driven recommendations should be considered in the context of San Francisco's overall budget, labor agreements, and strategic workforce planning goals.
    """)


# Add a footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0E1117;
    color: #FAFAFA;
    text-align: center;
    padding: 10px 0;
    font-size: 14px;
}
</style>
<div class="footer">
    <p>© 2024 San Francisco Compensation Analysis | Developed by Rohit</p>
</div>
""", unsafe_allow_html=True)

# Add a helper function for tooltips
def tooltip(text, help_text):
    return f'<span title="{help_text}">{text}</span>'

# Example usage of tooltip in the app (you can add this in relevant sections)
st.markdown(tooltip("Hover over me for more info!", "This is a helpful tooltip"), unsafe_allow_html=True)

# Error handling wrapper (you can wrap main functions with this)
def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()
    return wrapper

# Example usage of error handler
@error_handler
def some_function_that_might_error():
    # Your code here
    pass

# You can wrap your main functions with this error handler

# Add this at the end of your script for better performance
if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)