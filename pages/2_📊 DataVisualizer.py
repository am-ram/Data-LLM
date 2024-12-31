import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

def load_data():
    """Load and cache data"""
    data = st.file_uploader("Upload your dataset", type=["csv", "txt", "xls"], label_visibility="collapsed")
    if data is not None:
        return pd.read_csv(data)
    return None

def run_eda(df):
    """Run exploratory data analysis"""
    operations = {
        "Show Shape": lambda: st.write(df.shape),
        "Show Size": lambda: st.write(df.size),
        "Show Columns": lambda: st.write(df.columns.tolist()),
        "Select Columns": lambda: st.dataframe(df[st.multiselect("Select columns", df.columns, [])]),
        "Show Missing Values": lambda: st.write(df.isna().sum()),
        "Show Value Counts": lambda: st.write(df[st.selectbox("Select column", df.columns, index=None, placeholder="Choose a column")].value_counts()),
        "Show Summary": lambda: st.write(df.describe()),
        "Show Column Types": lambda: st.write(df[st.selectbox("Select column", df.columns, key="dtype", index=None, placeholder="Choose a column")].dtype)
    }
    
    for name, operation in operations.items():
        if st.checkbox(name):
            operation()

def create_plots(df):
    """Create various plots"""
    plot_options = {
        "Correlation Heatmap": lambda: plot_correlation(df),
        "Bar Graph": lambda: plot_bar(df),
        "Count Plot": lambda: plot_count(df),
        "Pie Chart": lambda: plot_pie(df),
        "Box Plot": lambda: plot_box(df),
        "Violin Plot": lambda: plot_violin(df),
        "Word Cloud": lambda: plot_wordcloud(df),
        "Time Series": lambda: plot_timeseries(df)
    }
    
    plot_type = st.selectbox("Select Plot Type", list(plot_options.keys()), index=None, placeholder="Choose a plot type")
    if plot_type:
        try:
            plot_options[plot_type]()
        except (ValueError, TypeError) as e:
            st.error("Invalid data selection for this plot type. Please select appropriate columns.")

def plot_correlation(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="Blues")
    st.pyplot(plt.gcf())
    plt.close()

def plot_bar(df):
    x_col = st.selectbox("Select X axis", df.columns, key="bar_x", index=None, placeholder="Choose X axis")
    y_col = st.selectbox("Select Y axis", df.columns, key="bar_y", index=None, placeholder="Choose Y axis")
    if x_col and y_col:
        fig = px.bar(df, x=x_col, y=y_col)
        st.plotly_chart(fig)

def plot_count(df):
    col = st.selectbox("Select column", df.columns, key="count", index=None, placeholder="Choose a column")
    if col:
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)

def plot_pie(df):
    col = st.selectbox("Select column", df.columns, key="pie", index=None, placeholder="Choose a column")
    if col:
        fig = px.pie(df, names=col, values=df[col].value_counts())
        st.plotly_chart(fig)

def plot_box(df):
    x_col = st.selectbox("Select X axis", df.columns, key="box_x", index=None, placeholder="Choose X axis")
    y_col = st.selectbox("Select Y axis", df.columns, key="box_y", index=None, placeholder="Choose Y axis")
    if x_col and y_col:
        fig = px.box(df, x=x_col, y=y_col)
        st.plotly_chart(fig)

def plot_violin(df):
    col = st.selectbox("Select column", df.columns, key="violin", index=None, placeholder="Choose a column")
    if col:
        fig = px.violin(df, y=col)
        st.plotly_chart(fig)

def plot_wordcloud(df):
    text_columns = df.select_dtypes(include=['object']).columns
    col = st.selectbox("Select text column", text_columns, key="wordcloud", index=None, placeholder="Choose a text column")
    if col:
        wordcloud = WordCloud().generate(' '.join(df[col].astype(str)))
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt.gcf())
        plt.close()

def plot_timeseries(df):
    col = st.selectbox("Select column", df.columns, key="timeseries", index=None, placeholder="Choose a column")
    if col:
        fig = px.line(df, y=col)
        st.plotly_chart(fig)

def generate_profile_report(df):
    """Generate pandas profiling report"""
    with st.spinner("Generating report..."):
        pr = ProfileReport(df, explorative=True)
        st_profile_report(pr)
        export = pr.to_html()
        st.download_button(
            label="Download Full Report",
            data=export,
            file_name='Report.html',
            mime='text/html'
        )

def main():
    st.title("Data Analysis Web App")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.write("This WebApp simplifies basic EDA and visualizations.")
    
    # Main content
    activities = ["EDA", "Plot", "In-Depth Report"]
    choice = st.sidebar.radio("Select Activity", activities, label_visibility="visible", index=None)
    
    df = load_data()
    if df is not None:
        st.dataframe(df)
        
        if choice == "EDA":
            st.header("Exploratory Data Analysis")
            run_eda(df)
        elif choice == "Plot":
            st.header("Data Visualization")
            create_plots(df)
        elif choice == "In-Depth Report":
            st.header("In-Depth Profile Report")
            generate_profile_report(df)

if __name__ == "__main__":
    main()