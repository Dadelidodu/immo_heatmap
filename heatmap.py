import streamlit as st
import os
import pandas as pd
import plotly.express as px
from map_display import *
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset using the cached function

st.set_page_config(page_title="Immo Belgium App", page_icon="🏠", layout="wide")

script_dir = os.path.dirname(os.path.abspath(__file__))
geojson_dataset_path = os.path.join(script_dir, 'json_files/gdf_merged.geojson')

_df = load_data(geojson_dataset_path)

normalized_dataset_path = os.path.join(script_dir, 'data/normalized_dataset.csv')
df_correlation = pd.read_csv(normalized_dataset_path)

df_correlation_wal = df_correlation[
    ((df_correlation['Zip Code'] >= 1300) & (df_correlation['Zip Code'] <= 1495)) | 
    ((df_correlation['Zip Code'] >= 4000) & (df_correlation['Zip Code'] <= 7999))]
df_correlation_fl = df_correlation[
    ((df_correlation['Zip Code'] >= 1500) & (df_correlation['Zip Code'] <= 3999)) | 
    ((df_correlation['Zip Code'] >= 8000) & (df_correlation['Zip Code'] <= 9999))]

df_correlation_bx = df_correlation[
    ((df_correlation['Zip Code'] >= 1000) & (df_correlation['Zip Code'] <= 1299))]

# Streamlit Interface

col2, col1 = st.columns([0.3, 0.7])

# Add selectbox for region selection with 'All regions' as the first option
with col2:  
    st.subheader("**Immo Belgium App**")
    select_reg = st.selectbox("Select a region", ['All regions', 'Wallonia', 'Flanders', 'Brussels'])
    select_chart = st.selectbox("Select a chart", ['Price Heatmap', 'Correlation Matrix'])
    
    if select_reg == 'All regions' and select_chart == 'Price Heatmap':
        with col1:
            st.markdown("""
            <style>
            iframe {
                width: 100% !important;
                height: 400px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            display_map(_df)
    
    elif select_reg and select_chart == 'Price Heatmap':

        if select_reg == 'Wallonia':

            # Filter the dataframe for the selected region
            _df_filtered = _df[_df['reg_name_fr'] == 'Région wallonne']
            
            with col2:

                select_mun = st.selectbox("Select a municipality", _df_filtered.sort_values(by='mun_name_fr',ascending=True)['mun_name_fr'].unique())
        
                if select_mun:
                    # Filter the dataset to get the selected municipality data
                    selected_municipality = _df_filtered[_df_filtered['mun_name_fr'] == select_mun].iloc[0]
                    
                    # Extract relevant details for the selected municipality
                    selected_postcode = selected_municipality['Zip Code']
                    selected_mean_price = selected_municipality['mean_price_by_locality']
                    
                    # Display the details in Streamlit
                    st.write(f"**Postal Code**: {selected_postcode}")
                    st.write(f"**Average Price per m² (€)**: {selected_mean_price:.2f}")
            
            # Clear previous map and create a new one in col1
            with col1:
                st.markdown("""
                <style>
                iframe {
                    width: 100% !important;
                    height: 400px !important;
                }
                </style>
                """, unsafe_allow_html=True)
                display_map(_df_filtered)
        
        
        if select_reg == 'Flanders':

            # Filter the dataframe for the selected region
            _df_filtered = _df[_df['reg_name_fr'] == 'Région flamande']
            
            with col2:

                select_mun = st.selectbox("Select a municipality", _df_filtered.sort_values(by='mun_name_fr',ascending=True)['mun_name_fr'].unique())
        
                if select_mun:
                    # Filter the dataset to get the selected municipality data
                    selected_municipality = _df_filtered[_df_filtered['mun_name_fr'] == select_mun].iloc[0]
                    
                    # Extract relevant details for the selected municipality
                    selected_postcode = selected_municipality['Zip Code']
                    selected_mean_price = selected_municipality['mean_price_by_locality']
                    
                    # Display the details in Streamlit
                    st.write(f"**Postal Code**: {selected_postcode}")
                    st.write(f"**Average Price per m² (€)**: {selected_mean_price:.2f}")
            
            # Clear previous map and create a new one in col1
            with col1:
                st.markdown("""
                <style>
                iframe {
                    width: 100% !important;
                    height: 400px !important;
                }
                </style>
                """, unsafe_allow_html=True)
                display_map(_df_filtered)
        
            
        if select_reg == 'Brussels':

            # Filter the dataframe for the selected region
            _df_filtered = _df[_df['reg_name_fr'] == 'Région de Bruxelles-Capitale']
            with col2:

                select_mun = st.selectbox("Select a municipality", _df_filtered.sort_values(by='mun_name_fr',ascending=True)['mun_name_fr'].unique())
        
                if select_mun:
                    # Filter the dataset to get the selected municipality data
                    selected_municipality = _df_filtered[_df_filtered['mun_name_fr'] == select_mun].iloc[0]
                    
                    # Extract relevant details for the selected municipality
                    selected_postcode = selected_municipality['Zip Code']
                    selected_mean_price = selected_municipality['mean_price_by_locality']
                    
                    # Display the details in Streamlit
                    st.write(f"**Postal Code**: {selected_postcode}")
                    st.write(f"**Average Price per m² (€)**: {selected_mean_price:.2f}")
            
            # Clear previous map and create a new one in col1
            with col1:
                st.markdown("""
                <style>
                iframe {
                    width: 100% !important;
                    height: 400px !important;
                }
                </style>
                """, unsafe_allow_html=True)
                display_map(_df_filtered)

    elif select_reg == 'All regions' and select_chart == 'Correlation Matrix':

        with col2:
            # Multiselect for selecting multiple columns to compute correlations
            select_columns = st.multiselect(
                "Select features to compare for correlation",
                ['Fully Equipped Kitchen', 'Furnished', 'Open Fire', 'Terrace', 'Garden', 
                'Swimming Pool', 'State of the Building Score', 
                'Primary Energy Consumption (kWh/m2) Score', 'Type of Property Score', 
                'Zip Code Score', 'Number of Rooms Score', 
                'Livable Space (m2) Score', 'Surface of the Land (m2) Score', 'Price'],
                default=['Fully Equipped Kitchen', 'Furnished', 'Open Fire', 'Terrace', 'Garden', 
                'Swimming Pool', 'State of the Building Score', 
                'Primary Energy Consumption (kWh/m2) Score', 'Type of Property Score', 
                'Zip Code Score', 'Number of Rooms Score', 
                'Livable Space (m2) Score', 'Surface of the Land (m2) Score', 'Price']  # Provide sensible defaults
            )
        
        with col1:

            if len(select_columns) > 1:  # Ensure at least two columns are selected
                # Compute the correlation matrix
                correlation_matrix = df_correlation[select_columns].corr()

                # Create an interactive heatmap using Plotly without annotations or legend
                fig = px.imshow(
                    correlation_matrix,
                    color_continuous_scale=[[0, '#202060'], [1, '#00ff00']],  # Custom color scale
                    text_auto=".2f"
                )

                # Add layout settings (no width/height here)
                fig.update_layout(
                    width=800,
                    height=800,
                    xaxis=dict(
                        showticklabels=False,  # Remove x-axis labels (column names)
                        ticks='',              # Remove x-axis ticks
                        title='',              # Remove x-axis title
                    ),
                    yaxis=dict(
                        showticklabels=False,  # Remove y-axis labels (row names)
                        ticks='',              # Remove y-axis ticks
                        title='',              # Remove y-axis title
                    ),
                    coloraxis_showscale=False  # Hide the color bar (legend)
                )

                # Render the Plotly heatmap in Streamlit

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.write("Please select at least two features to compute the correlation matrix.")

    
    elif select_reg and select_chart == 'Correlation Matrix':

        if select_reg == 'Wallonia':

        
            with col2:
                # Multiselect for selecting multiple columns to compute correlations
                select_columns = st.multiselect(
                    "Select features to compare for correlation",
                    ['Fully Equipped Kitchen', 'Furnished', 'Open Fire', 'Terrace', 'Garden', 
                    'Swimming Pool', 'State of the Building Score', 
                    'Primary Energy Consumption (kWh/m2) Score', 'Type of Property Score', 
                    'Zip Code Score', 'Number of Rooms Score', 
                    'Livable Space (m2) Score', 'Surface of the Land (m2) Score', 'Price'],
                    default=['Fully Equipped Kitchen', 'Furnished', 'Open Fire', 'Terrace', 'Garden', 
                    'Swimming Pool', 'State of the Building Score', 
                    'Primary Energy Consumption (kWh/m2) Score', 'Type of Property Score', 
                    'Zip Code Score', 'Number of Rooms Score', 
                    'Livable Space (m2) Score', 'Surface of the Land (m2) Score', 'Price']  # Provide sensible defaults
                )
            
            with col1:
                if len(select_columns) > 1:  # Ensure at least two columns are selected
                    # Compute the correlation matrix
                    correlation_matrix = df_correlation_wal[select_columns].corr()

                    # Create an interactive heatmap using Plotly without annotations or legend
                    fig = px.imshow(
                    correlation_matrix,
                    color_continuous_scale=[[0, '#202060'], [1, '#00ff00']],  # Custom color scale
                    text_auto=".2f")

                    # Add titles and update layout
                    fig.update_layout(
                    width=800,
                    height=800,
                    xaxis=dict(
                        showticklabels=False,  # Remove x-axis labels (column names)
                        ticks='',              # Remove x-axis ticks
                        title='',              # Remove x-axis title
                    ),
                    yaxis=dict(
                        showticklabels=False,  # Remove y-axis labels (row names)
                        ticks='',              # Remove y-axis ticks
                        title='',              # Remove y-axis title
                    ),
                    coloraxis_showscale=False  # Hide the color bar (legend)
                    )
                    # Render the Plotly heatmap in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Please select at least two features to compute the correlation matrix.")

        if select_reg == 'Flanders':

        
            with col2:
                # Multiselect for selecting multiple columns to compute correlations
                select_columns = st.multiselect(
                    "Select features to compare for correlation",
                    ['Fully Equipped Kitchen', 'Furnished', 'Open Fire', 'Terrace', 'Garden', 
                    'Swimming Pool', 'State of the Building Score', 
                    'Primary Energy Consumption (kWh/m2) Score', 'Type of Property Score', 
                    'Zip Code Score', 'Number of Rooms Score', 
                    'Livable Space (m2) Score', 'Surface of the Land (m2) Score', 'Price'],
                    default=['Fully Equipped Kitchen', 'Furnished', 'Open Fire', 'Terrace', 'Garden', 
                    'Swimming Pool', 'State of the Building Score', 
                    'Primary Energy Consumption (kWh/m2) Score', 'Type of Property Score', 
                    'Zip Code Score', 'Number of Rooms Score', 
                    'Livable Space (m2) Score', 'Surface of the Land (m2) Score', 'Price']  # Provide sensible defaults
                )
            
            with col1:
                if len(select_columns) > 1:  # Ensure at least two columns are selected
                    # Compute the correlation matrix
                    correlation_matrix = df_correlation_fl[select_columns].corr()

                    # Create an interactive heatmap using Plotly without annotations or legend
                    fig = px.imshow(
                    correlation_matrix,
                    color_continuous_scale=[[0, '#202060'], [1, '#00ff00']],  # Custom color scale
                    text_auto=".2f")

                    # Add titles and update layout
                    fig.update_layout(
                    width=800,
                    height=800,
                    xaxis=dict(
                        showticklabels=False,  # Remove x-axis labels (column names)
                        ticks='',              # Remove x-axis ticks
                        title='',              # Remove x-axis title
                    ),
                    yaxis=dict(
                        showticklabels=False,  # Remove y-axis labels (row names)
                        ticks='',              # Remove y-axis ticks
                        title='',              # Remove y-axis title
                    ),
                    coloraxis_showscale=False  # Hide the color bar (legend)
                    )

                    # Render the Plotly heatmap in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Please select at least two features to compute the correlation matrix.")
        
        if select_reg == 'Brussels':

        
            with col2:
                # Multiselect for selecting multiple columns to compute correlations
                select_columns = st.multiselect(
                    "Select features to compare for correlation",
                    ['Fully Equipped Kitchen', 'Furnished', 'Open Fire', 'Terrace', 'Garden', 
                    'Swimming Pool', 'State of the Building Score', 
                    'Primary Energy Consumption (kWh/m2) Score', 'Type of Property Score', 
                    'Zip Code Score', 'Number of Rooms Score', 
                    'Livable Space (m2) Score', 'Surface of the Land (m2) Score', 'Price'],
                    default=['Fully Equipped Kitchen', 'Furnished', 'Open Fire', 'Terrace', 'Garden', 
                    'Swimming Pool', 'State of the Building Score', 
                    'Primary Energy Consumption (kWh/m2) Score', 'Type of Property Score', 
                    'Zip Code Score', 'Number of Rooms Score', 
                    'Livable Space (m2) Score', 'Surface of the Land (m2) Score', 'Price']  # Provide sensible defaults
                )
            
            with col1:
                if len(select_columns) > 1:  # Ensure at least two columns are selected
                    # Compute the correlation matrix
                    correlation_matrix = df_correlation_bx[select_columns].corr()

                    # Create an interactive heatmap using Plotly without annotations or legend
                    fig = px.imshow(
                    correlation_matrix,
                    color_continuous_scale=[[0, '#202060'], [1, '#00ff00']],  # Custom color scale
                    text_auto=".2f")
                    
                    # Add titles and update layout
                    fig.update_layout(
                    width=800,
                    height=800,
                    xaxis=dict(
                        showticklabels=False,  # Remove x-axis labels (column names)
                        ticks='',              # Remove x-axis ticks
                        title='',              # Remove x-axis title
                    ),
                    yaxis=dict(
                        showticklabels=False,  # Remove y-axis labels (row names)
                        ticks='',              # Remove y-axis ticks
                        title='',              # Remove y-axis title
                    ),
                    coloraxis_showscale=False  # Hide the color bar (legend)
                    )

                    # Render the Plotly heatmap in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Please select at least two features to compute the correlation matrix.")

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    # Load data
    data = pd.read_csv("data/normalized_dataset.csv")

    # Define predictor columns and target column
    X = data[['Zip Code Score', 'Type of Property Score', 'Number of Rooms Score', 'Livable Space (m2) Score', 'Surface of the Land (m2) Score']]
    y = data['Price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    return data, X_train, X_test, y_train, y_test

with col2:

    # Train Linear Regression model
    @st.cache_resource
    def train_model(X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    # Find nearest Zip Code within ±5
    def find_nearest_zip_code(zip_code, zip_code_scores):
        valid_zip_codes = zip_code_scores.index
        nearby_zip_codes = valid_zip_codes[(valid_zip_codes >= zip_code - 5) & (valid_zip_codes <= zip_code + 5)]
        if not nearby_zip_codes.empty:
            return nearby_zip_codes[0]  # Return the first valid Zip Code
        else:
            return None  # No matching Zip Code found

    # Load and preprocess data
    data, X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train the Linear Regression model
    lr_model = train_model(X_train, y_train)

    # Evaluate the model
    r2_score = lr_model.score(X_test, y_test)

    # Streamlit app interface
    st.subheader("Price Predictor")

    # User inputs
    zip_code = st.number_input("Enter Zip Code", min_value=1000, max_value=9999, step=1)
    prop_type = st.selectbox("Select Type of Property", options=data['Type of Property'].unique())
    livable_space = st.number_input("Enter Livable Space (m2)", min_value=10.0, step=1.0)
    land_area = st.number_input("Enter Surface of the Land (m2)", min_value=0.0, step=1.0)
    rooms_number = st.number_input("Enter Number of Rooms", min_value=0.0, step=1.0)

    if st.button("Predict Price"):
        # Calculate scores for inputs
        zip_code_scores = data.groupby('Zip Code')['Price Ratio'].median()
        nearest_zip_code = find_nearest_zip_code(zip_code, zip_code_scores)
        if nearest_zip_code is not None:
            zip_score = zip_code_scores[nearest_zip_code]
            st.write(f"Using nearest Zip Code: {nearest_zip_code}")
        else:
            zip_score = zip_code_scores.median()
            st.write("No nearby Zip Code found. Using median Zip Code score.")

        prop_type_scores = data.groupby('Type of Property')['Price Ratio'].mean()
        building_scores = data.groupby('State of the Building')['Price Ratio'].mean()
        livable_space_scores = data.groupby('Livable Space (m2)')['Price Ratio'].median()
        land_scores = data.groupby('Surface of the Land (m2)')['Price Ratio'].median()
        room_scores = data.groupby('Number of Rooms')['Price Ratio'].mean()

        # Normalize scores
        zip_score /= zip_code_scores.max()
        prop_type_score = prop_type_scores.get(prop_type, 0) / prop_type_scores.max()
        livable_space_score = livable_space_scores.get(livable_space, 0) / livable_space_scores.max()
        land_score = land_scores.get(land_area, 0) / land_scores.max()
        rooms_score = room_scores.get(rooms_number, 0) / room_scores.max()

        # Prepare input features
        input_features = np.array([[zip_score, prop_type_score, rooms_score, livable_space_score, land_score]])

        # Predict using the trained model
        predicted_price = lr_model.predict(input_features)[0]

        # Display the prediction
        st.write(f"### Predicted Price: €{predicted_price:,.2f}")
with col1:
    st.write(df_correlation)
