import streamlit as st
import requests
from datetime import datetime

# Function to get the user's IP address using an external API (ipify)
def get_ip():
    try:
        # Get the IP address of the user from the ipify API
        response = requests.get('https://api.ipify.org?format=json')
        ip = response.json()['ip']
    except requests.exceptions.RequestException as e:
        # If there's an issue (e.g., no internet connection), return a fallback message
        ip = "Unable to fetch IP"
    return ip

# Function to log the IP address to a file
def log_ip(ip):
    # Open a text file in append mode
    with open("ip_log.txt", "a") as log_file:
        # Write the timestamp and IP address to the log file
        log_file.write(f"{datetime.now()} - IP Address: {ip}\n")

# Get the user's IP address
user_ip = get_ip()

# Log the IP address to the file
log_ip(user_ip)

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Home",
    page_icon="👋",
    layout="wide"
)

# Main content of the app
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
