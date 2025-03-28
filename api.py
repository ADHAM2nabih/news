import streamlit as st
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import pandas as pd

# Category mapping dictionary
CATEGORY_MAP = {
    0: 'ARTS',
    1: 'ARTS & CULTURE',
    2: 'BLACK VOICES',
    3: 'BUSINESS',
    4: 'COLLEGE',
    5: 'COMEDY',
    6: 'CRIME',
    7: 'EDUCATION',
    8: 'ENTERTAINMENT',
    9: 'FIFTY',
    10: 'GOOD NEWS',
    11: 'GREEN',
    12: 'HEALTHY LIVING',
    13: 'IMPACT',
    14: 'LATINO VOICES',
    15: 'MEDIA',
    16: 'PARENTS',
    17: 'POLITICS',
    18: 'QUEER VOICES',
    19: 'RELIGION',
    20: 'SCIENCE',
    21: 'SPORTS',
    22: 'STYLE',
    23: 'TASTE',
    24: 'TECH',
    25: 'THE WORLDPOST',
    26: 'TRAVEL',
    27: 'WEIRD NEWS',
    28: 'WOMEN',
    29: 'WORLD NEWS',
    30: 'WORLDPOST'
}

# Initialize database
def init_db():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  text TEXT,
                  prediction INTEGER,
                  category TEXT,
                  timestamp DATETIME)''')
    conn.commit()
    conn.close()

# Save prediction to database
def save_feedback(text, prediction):
    category = CATEGORY_MAP.get(prediction, f"Unknown Category ({prediction})")
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute("INSERT INTO predictions (text, prediction, category, timestamp) VALUES (?, ?, ?, ?)",
              (text, prediction, category, datetime.now()))
    conn.commit()
    conn.close()

# Get all predictions from database
def get_all_predictions():
    conn = sqlite3.connect('feedback.db')
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# Load the model
@st.cache_resource
def load_model():
    try:
        return joblib.load('news_classifier_pipeline.joblib')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Prediction function
def predict_category(text):
    model = load_model()
    if model is None:
        return None
    
    prediction = model.predict([text])[0]
    return int(prediction)

# Streamlit app
def main():
    # Initialize database
    init_db()
    
    # Set page configuration
    st.set_page_config(
        page_title="News Category Classifier",
        page_icon="📰",
        layout="centered"
    )

    # Custom CSS
    st.markdown("""
    <style>
    :root {
        --primary: #4361ee;
        --secondary: #3f37c9;
        --accent: #4895ef;
        --light: #f8f9fa;
        --dark: #212529;
        --success: #4cc9f0;
        --warning: #f72585;
        --error: #f72585;
    }
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #142238 0%, #c3cfe2 100%);
        min-height: 100vh;
        padding: 2rem;
        color: var(--dark);
    }
    
    .container {
        max-width: 800px;
        margin: 0 auto;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }
    
    header {
        background: linear-gradient(to right, var(--primary), var(--secondary));
        color: white;
        padding: 2rem;
        text-align: center;
    }
    
    h1 {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-weight: 300;
        opacity: 0.9;
    }
    
    .content {
        padding: 2rem;
    }
    
    .input-group {
        margin-bottom: 2rem;
    }
    
    label {
        display: block;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    textarea {
        width: 100%;
        padding: 1rem;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        font-family: inherit;
        font-size: 1rem;
        resize: vertical;
        min-height: 150px;
        transition: border 0.3s;
    }
    
    textarea:focus {
        outline: none;
        border-color: var(--accent);
    }
    
    button {
        background: linear-gradient(to right, var(--primary), var(--secondary));
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1rem;
        border-radius: 8px;
        cursor: pointer;
        font-family: inherit;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        width: 100%;
        margin-top: 1rem;
    }
    
    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    button:active {
        transform: translateY(0);
    }
    
    button:disabled {
        background: #cccccc;
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
    }
    
    .result {
        margin-top: 2rem;
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        display: none;
        animation: fadeIn 0.5s;
    }
    
    .result.show {
        display: block;
    }
    
    .result h3 {
        color: var(--primary);
        margin-bottom: 0.5rem;
    }
    
    .category {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--secondary);
        margin-top: 0.5rem;
        text-transform: capitalize;
    }
    
    .error {
        color: var(--error);
        margin-top: 1rem;
        font-weight: 500;
    }
    
    .loading {
        display: none;
        text-align: center;
        margin: 1rem 0;
    }
    
    .spinner {
        border: 4px solid rgba(0, 0, 0, 0.1);
        border-radius: 50%;
        border-top: 4px solid var(--primary);
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    .big-font {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--primary);
    }
    
    .category-font {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--secondary);
        padding: 0.5rem;
        background-color: #f0f5ff;
        border-radius: 8px;
        margin-top: 0.5rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    footer {
        text-align: center;
        margin-top: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    @media (max-width: 600px) {
        body {
            padding: 1rem;
        }
        
        h1 {
            font-size: 1.8rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("📰 News Category Classifier")
    st.markdown("Discover the category of your news article using AI!")

    # Text input
    news_text = st.text_area(
        "Enter News Text:", 
        placeholder="Paste your news article here...", 
        height=250
    )

    # Prediction button
    if st.button("Classify Article", type="primary"):
        # Validate input
        if not news_text.strip():
            st.error("Please enter some news text")
        else:
            # Show loading spinner
            with st.spinner("Analyzing your news article..."):
                # Make prediction
                prediction = predict_category(news_text)
                
                # Display result
                if prediction is not None:
                    category = CATEGORY_MAP.get(prediction, f"Unknown Category ({prediction})")
                    
                    # Save to database
                    save_feedback(news_text, prediction)
                    
                    # Result display with custom styling
                    st.markdown('<p class="big-font">Predicted Category:</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="category-font">{category}</p>', unsafe_allow_html=True)
                else:
                    st.error("Failed to make a prediction. Please check the model.")

    # Feedback history section
    st.markdown("---")
    if st.button("View Prediction History"):
        st.subheader("📋 Prediction History")
        
        # Get all predictions from database
        df = get_all_predictions()
        
        if df.empty:
            st.info("No predictions have been made yet.")
        else:
            # Show statistics
            col1, col2 = st.columns(2)
            col1.metric("Total Predictions", len(df))
            most_common = df['category'].mode()[0]
            col2.metric("Most Common Category", most_common)
            
            # Show data table
            st.dataframe(
                df[['timestamp', 'category', 'text']].rename(columns={
                    'timestamp': 'Time',
                    'category': 'Category',
                    'text': 'Text Preview'
                }),
                column_config={
                    "Text Preview": st.column_config.TextColumn(
                        "Text Preview",
                        width="medium",
                        help="The text that was classified",
                    ),
                    "Time": st.column_config.DatetimeColumn(
                        "Time",
                        format="YYYY-MM-DD HH:mm:ss",
                    ),
                    "Category": st.column_config.TextColumn(
                        "Category",
                        width="small",
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Data as CSV",
                data=csv,
                file_name='news_classification_history.csv',
                mime='text/csv',
            )

    # Footer
    st.markdown("---")
    st.markdown("*AI-powered News Category Classification*")

# Run the app
if __name__ == "__main__":
    main()