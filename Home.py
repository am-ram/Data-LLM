import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="👋",
    layout="wide"
)

st.write("# Welcome to DataInsights AI! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Welcome to an all-in-one data analysis platform that combines the power of Generative AI, 
    interactive visualizations, and machine learning to help you derive meaningful insights from your data.
    
    ### 💬  Chat with Your Data (Page 1)
    Upload your CSV file and interact with it using a state-of-the-art LLaMA-8b language model. Ask questions 
    about your data in natural language and get instant insights - from simple queries to complex analysis.
    
    ### 📊 Interactive Visualizations (Page 2)
    Transform your data into compelling visualizations with just a few clicks:
    - Create various types of plots and charts
    - Customize visualizations by selecting specific columns
    - Generate interactive and exportable visualization
    
    ### 🎯 ML Model Training & Evaluation (Page 3)
    Build and evaluate machine learning models without writing code:
    - Choose between classification and regression tasks
    - Select from multiple algorithms (Random Forest, SVM, etc.)
    - Get instant performance metrics and visualizations
    
    **👈 Select an analysis type from the sidebar** to begin exploring your data!
    
    Need help? Each page includes tooltips and explanations to guide you through the process.
    """
)