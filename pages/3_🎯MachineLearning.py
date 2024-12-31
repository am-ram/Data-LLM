import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
import plotly.express as px

def plot_results(y_test, predictions, problem_type):
    """Plot actual vs predicted values"""
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions
    })
    
    if problem_type == 'Regression':
        fig = px.scatter(results_df, x='Actual', y='Predicted', 
                        title='Actual vs Predicted Values')
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=y_test.min(), y0=y_test.min(),
            x1=y_test.max(), y1=y_test.max()
        )
        st.plotly_chart(fig)
        
        # Add residuals plot
        residuals = y_test - predictions
        fig_residuals = px.scatter(x=predictions, y=residuals,
                                 labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                 title='Residuals Plot')
        fig_residuals.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_residuals)
    else:
        # For classification, create a confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_test, predictions)
        fig = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual"),
                       title='Confusion Matrix',
                       color_continuous_scale='Blues')
        st.plotly_chart(fig)
        
        # Add prediction distribution
        fig_dist = px.histogram(results_df, x=['Actual', 'Predicted'], 
                              barmode='group',
                              title='Distribution of Actual vs Predicted Classes')
        st.plotly_chart(fig_dist)

def display_metrics(metrics, problem_type):
    """Display metrics using native Streamlit components"""
    if problem_type == 'Classification':
        # Create three columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Accuracy Score",
                value=f"{metrics['accuracy']:.2%}"
            )
        
        # Parse classification report for precision and recall
        report_dict = classification_report(metrics['y_test'], metrics['predictions'], output_dict=True)
        avg_precision = report_dict['weighted avg']['precision']
        avg_recall = report_dict['weighted avg']['recall']
        
        with col2:
            st.metric(
                label="Precision (Weighted)",
                value=f"{avg_precision:.2%}"
            )
        
        with col3:
            st.metric(
                label="Recall (Weighted)",
                value=f"{avg_recall:.2%}"
            )
        
        # Detailed classification report in an expander
        with st.expander("View Detailed Classification Report"):
            st.text(metrics['report'])
            
    else:  # Regression
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="R² Score",
                value=f"{metrics['r2']:.4f}",
                help="Closer to 1 is better"
            )
        
        with col2:
            st.metric(
                label="Root Mean Squared Error",
                value=f"{np.sqrt(metrics['mse']):.4f}",
                help="Lower is better"
            )
        
        # Additional regression metrics in an expander
        with st.expander("View Additional Metrics"):
            st.write(f"Mean Squared Error: {metrics['mse']:.4f}")
            st.write(f"Mean Absolute Error: {metrics['mae']:.4f}")
            
def train_model(model, X_train, X_test, y_train, y_test, problem_type):
    """Train model and return metrics"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if problem_type == 'Classification':
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return {
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred,
            'y_test': y_test  # Added for classification report parsing
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        return {
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'predictions': y_pred
        }

def main():
    st.title("Machine Learning Web App")
    
    with st.sidebar:
        st.header("Model Configuration")
        st.markdown("""
        This app helps you:
        - Train ML models on your data
        - Evaluate model performance
        - Visualize predictions
        """)
    
    # Load data
    data = st.file_uploader("Upload your dataset", type=["csv", "txt", "xls"])
    
    if data is not None:
        df = pd.read_csv(data)
        
        # Display sample data in an expander
        with st.expander("Preview Dataset"):
            st.dataframe(df.head())
            st.write("Dataset Shape:", df.shape)
        
        # Select problem type
        problem_type = st.selectbox(
            "Select Problem Type",
            ["Classification", "Regression"],
            index=None,
            placeholder="Choose problem type"
        )
        
        if problem_type:
            col1, col2 = st.columns(2)
            
            with col1:
                feature_cols = st.multiselect(
                    "Select Feature Columns",
                    df.columns,
                    placeholder="Choose features for training"
                )
            
            with col2:
                target_col = st.selectbox(
                    "Select Target Column",
                    [col for col in df.columns if col not in feature_cols],
                    index=None,
                    placeholder="Choose target variable"
                )
            
            if feature_cols and target_col:
                # Model selection
                models = {
                    'Classification': {
                        'Logistic Regression': LogisticRegression(),
                        'KNN': KNeighborsClassifier(),
                        'SVM': SVC(),
                        'Random Forest': RandomForestClassifier()
                    },
                    'Regression': {
                        'Linear Regression': LinearRegression(),
                        'Random Forest': RandomForestRegressor()
                    }
                }
                
                with st.sidebar:
                    selected_model = st.selectbox(
                        "Select Model",
                        list(models[problem_type].keys()),
                        index=None,
                        placeholder="Choose a model"
                    )
                    
                    if selected_model:
                        st.write("---")
                        st.write("Model Parameters")
                        
                        if selected_model == 'KNN':
                            n_neighbors = st.slider('Number of neighbors', 1, 20, 5)
                            model = KNeighborsClassifier(n_neighbors=n_neighbors)
                        elif selected_model == 'Random Forest':
                            n_estimators = st.slider('Number of trees', 10, 100, 50)
                            max_depth = st.slider('Maximum depth', 1, 50, 10)
                            model = (RandomForestClassifier if problem_type == 'Classification' 
                                   else RandomForestRegressor)(n_estimators=n_estimators, max_depth=max_depth)
                        else:
                            model = models[problem_type][selected_model]
                
                if selected_model:
                    if st.button('Train Model', type='primary'):
                        with st.spinner('Training model...'):
                            # Prepare data
                            X = df[feature_cols]
                            y = df[target_col]
                            X = pd.get_dummies(X)  # Handle categorical variables
                            
                            # Scale features
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_scaled, y, test_size=0.2, random_state=42
                            )
                            
                            # Train and evaluate
                            metrics = train_model(
                                model, X_train, X_test, y_train, y_test, problem_type
                            )
                            
                            st.success('Model training completed!')
                            
                            # Results section
                            st.write("---")
                            st.subheader("Model Performance")
                            display_metrics(metrics, problem_type)
                            
                            # Visualizations
                            st.write("---")
                            st.subheader("Predictions Visualization")
                            plot_results(y_test, metrics['predictions'], problem_type)
                            
                            # Feature importance for Random Forest
                            if selected_model == 'Random Forest':
                                st.write("---")
                                st.subheader("Feature Importance")
                                importance_df = pd.DataFrame({
                                    'Feature': X.columns,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(
                                    importance_df,
                                    x='Feature',
                                    y='Importance',
                                    title='Feature Importance Plot'
                                )
                                st.plotly_chart(fig)

if __name__ == "__main__":
    main()