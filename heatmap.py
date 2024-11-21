import folium
import branca
import streamlit as st
import geopandas as gpd
import os
from streamlit_folium import st_folium
import pandas as pd
import plotly.express as px

# Load the GeoDataFrame + Correlation DataFrame
script_dir = os.path.dirname(os.path.abspath(__file__))
geojson_dataset_path = os.path.join(script_dir, 'json_files/geojson_dataset.geojson')

@st.cache_data
def load_data():
    return gpd.read_file(geojson_dataset_path)

# Load the dataset using the cached function
df = load_data()
df['postcode'] = df['postcode'].astype(int)
df['mean_price'] = df['mean_price'].astype(int)

normalized_dataset_path = os.path.join(script_dir, 'data/normalized_dataset.csv')
df_correlation = pd.read_csv(normalized_dataset_path)

df_correlation_wal = df_correlation[
    ((df_correlation['Zip Code'] >= 1300) & (df_correlation['Zip Code'] <= 1495)) | 
    ((df_correlation['Zip Code'] >= 4000) & (df_correlation['Zip Code'] <= 7999))
]
df_correlation_fl = df_correlation[
    ((df_correlation['Zip Code'] >= 1500) & (df_correlation['Zip Code'] <= 3999)) | 
    ((df_correlation['Zip Code'] >= 8000) & (df_correlation['Zip Code'] <= 9999))
]

df_correlation_bx = df_correlation[
    ((df_correlation['Zip Code'] >= 1000) & (df_correlation['Zip Code'] <= 1299))
]

# Streamlit Interface

# Create two columns in the Streamlit layout
col1, col2 = st.columns([1.3, 2.7])

# Add selectbox for region selection with 'All regions' as the first option
with col1:
    st.subheader("**Immo Belgium App**")
    select_reg = st.selectbox("Select a region", ['All regions'] + ['Wallonia','Flanders','Brussels'])
    select_chart = st.selectbox("Select a chart", ['Price/m2 by Postcode', 'Correlation Matrix'])

    if select_reg == 'All regions' and select_chart == 'Price/m2 by Postcode':

        with col1:

               # Extract relevant details for the selected municipality
                selected_postcode = int(len(df['postcode']))
                selected_mean_price = int(df['mean_price'].mean())
                
                # Display the details in Streamlit
                st.write(f"**Number of Postcodes**: {selected_postcode:.2f}")
                st.write(f"**Average Price per m² (€)**: {selected_mean_price:.2f}")
        
        with col2:
            # Create the folium map, centered at Belgium
            m = folium.Map(location=[50.8503, 4.3517], zoom_start=7)  
            folium.TileLayer('cartodb positron').add_to(m)  # 'cartodb positron' is a light, grayscale map

            # Create the colormap using `branca` library
            branca_cmap = branca.colormap.linear.YlOrRd_09.scale(df['mean_price'].min(), df['mean_price'].max())

            # Function to handle clicks on each feature (municipality)
            def on_each_feature(feature, layer):
                # Extract the unique identifier from the feature
                feature_postcode = feature['properties']['postcode']

                # Find the corresponding row in the DataFrame using the postcode
                matching_row = df[df['postcode'] == feature_postcode].iloc[0]  # Assuming each postcode is unique

                # Extract the details from the matching row in the DataFrame
                municipality_name = matching_row['mun_name_fr']
                postcode = matching_row['postcode']
                mean_price = matching_row['mean_price']

                # Display the details in Streamlit when the feature is clicked
                st.write(f"**Municipality**: {municipality_name}")
                st.write(f"**Postal Code**: {postcode}")
                st.write(f"**Average Price per m² (€)**: {mean_price:.2f}")

            # Create GeoJson layer
            geojson_layer = folium.GeoJson(
                df,  # Assuming df is GeoJSON formatted data
                style_function=lambda feature: {
                    'fillColor': branca_cmap(feature['properties']['mean_price']),
                    'color': 'white',
                    'weight': 0.3,
                    'fillOpacity': 1,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['postcode', 'mean_price', 'mun_name_fr'],
                    aliases=['Postal Code:', 'Price/m² (€):', 'Municipality:'],
                    localize=True,
                ),
            )

            # Set on_each_feature function to the GeoJson layer
            geojson_layer.add_child(folium.GeoJsonPopup(fields=['mun_name_fr', 'postcode', 'mean_price']))
            geojson_layer.on_each_feature = on_each_feature  # Ensure this is properly assigned

            # Add the GeoJson layer to the map
            geojson_layer.add_to(m)

            # Add the colormap legend to the map
            branca_cmap.add_to(m)

            # Render the Folium map with Streamlit
            st_folium(m, height=400, use_container_width=True)

    elif select_reg and select_chart == 'Price/m2 by Postcode':

        if select_reg == 'Wallonia':
            # Filter the dataframe for the selected region
            df_filtered = df[df['reg_name_fr'] == 'Région wallonne']
            with col1:

                select_mun = st.selectbox("Select a municipality", df_filtered.sort_values(by='mun_name_fr',ascending=True)['mun_name_fr'].unique())
        
                if select_mun:
                    # Filter the dataset to get the selected municipality data
                    selected_municipality = df_filtered[df_filtered['mun_name_fr'] == select_mun].iloc[0]
                    
                    # Extract relevant details for the selected municipality
                    selected_postcode = selected_municipality['postcode']
                    selected_mean_price = selected_municipality['mean_price']
                    
                    # Display the details in Streamlit
                    st.write(f"**Postal Code**: {selected_postcode}")
                    st.write(f"**Average Price per m² (€)**: {selected_mean_price:.2f}")
            
            # Clear previous map and create a new one in col2
            with col2:
                # Create the folium map, centered at Belgium
                m = folium.Map(location=[50.8503, 4.3517], zoom_start=7)  
                
                folium.TileLayer('cartodb positron').add_to(m)  # 'cartodb positron' is a light, grayscale map

                # Create the colormap using `branca` library
                branca_cmap = branca.colormap.linear.YlOrRd_09.scale(df_filtered['mean_price'].min(), df_filtered['mean_price'].max())
                
                # Add GeoJson layer with custom colormap
                folium.GeoJson(
                    df_filtered,
                    style_function=lambda feature: {
                        'fillColor': branca_cmap(feature['properties']['mean_price']),
                        'color': 'white',
                        'weight': 0.3,
                        'fillOpacity': 1,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['postcode', 'mean_price', 'mun_name_fr'],
                        aliases=['Postal Code:', 'Price/m² (€):', 'Municipality:'],
                        localize=True,
                    ),
                ).add_to(m)

                # Add the colormap legend to the map
                branca_cmap.add_to(m)

                # Render the folium map in Streamlit
                st_folium(m, height=400, use_container_width=True)
        
        if select_reg == 'Flanders':
            # Filter the dataframe for the selected region
            df_filtered = df[df['reg_name_fr'] == 'Région flamande']
            with col1:

                select_mun = st.selectbox("Select a municipality", df_filtered.sort_values(by='mun_name_fr',ascending=True)['mun_name_fr'].unique())
        
                if select_mun:
                    # Filter the dataset to get the selected municipality data
                    selected_municipality = df_filtered[df_filtered['mun_name_fr'] == select_mun].iloc[0]
                    
                    # Extract relevant details for the selected municipality
                    selected_postcode = selected_municipality['postcode']
                    selected_mean_price = selected_municipality['mean_price']
                    
                    # Display the details in Streamlit
                    st.write(f"**Postal Code**: {selected_postcode}")
                    st.write(f"**Average Price per m² (€)**: {selected_mean_price:.2f}")
            
            # Clear previous map and create a new one in col2
            with col2:
                # Create the folium map, centered at Belgium
                m = folium.Map(location=[50.8503, 4.3517], zoom_start=7)  
                folium.TileLayer('cartodb positron').add_to(m)  # 'cartodb positron' is a light, grayscale map

                # Create the colormap using `branca` library
                branca_cmap = branca.colormap.linear.YlOrRd_09.scale(df_filtered['mean_price'].min(), df_filtered['mean_price'].max())
                
                # Add GeoJson layer with custom colormap
                folium.GeoJson(
                    df_filtered,
                    style_function=lambda feature: {
                        'fillColor': branca_cmap(feature['properties']['mean_price']),
                        'color': 'white',
                        'weight': 0.3,
                        'fillOpacity': 1,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['postcode', 'mean_price', 'mun_name_fr'],
                        aliases=['Postal Code:', 'Price/m² (€):', 'Municipality:'],
                        localize=True,
                    ),
                ).add_to(m)

                # Add the colormap legend to the map
                branca_cmap.add_to(m)

                # Render the folium map in Streamlit
                st_folium(m, height=400, use_container_width=True)

        if select_reg == 'Brussels':
            # Filter the dataframe for the selected region
            df_filtered = df[df['reg_name_fr'] == 'Région de Bruxelles-Capitale']
            with col1:

                select_mun = st.selectbox("Select a municipality", df_filtered.sort_values(by='mun_name_fr',ascending=True)['mun_name_fr'].unique())
        
                if select_mun:
                    # Filter the dataset to get the selected municipality data
                    selected_municipality = df_filtered[df_filtered['mun_name_fr'] == select_mun].iloc[0]
                    
                    # Extract relevant details for the selected municipality
                    selected_postcode = selected_municipality['postcode']
                    selected_mean_price = selected_municipality['mean_price']
                    
                    # Display the details in Streamlit
                    st.write(f"**Postal Code**: {selected_postcode}")
                    st.write(f"**Average Price per m² (€)**: {selected_mean_price:.2f}")
            
            # Clear previous map and create a new one in col2
            with col2:
                # Create the folium map, centered at Belgium
                m = folium.Map(location=[50.8503, 4.3517], zoom_start=7) 
                
                folium.TileLayer('cartodb positron').add_to(m)  # 'cartodb positron' is a light, grayscale map

                # Create the colormap using `branca` library
                branca_cmap = branca.colormap.linear.YlOrRd_09.scale(df_filtered['mean_price'].min(), df_filtered['mean_price'].max())
                
                # Add GeoJson layer with custom colormap
                folium.GeoJson(
                    df_filtered,
                    style_function=lambda feature: {
                        'fillColor': branca_cmap(feature['properties']['mean_price']),
                        'color': 'white',
                        'weight': 0.3,
                        'fillOpacity': 1,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['postcode', 'mean_price', 'mun_name_fr'],
                        aliases=['Postal Code:', 'Price/m² (€):', 'Municipality:'],
                        localize=True,
                    ),
                ).add_to(m)

                # Add the colormap legend to the map
                branca_cmap.add_to(m)

                # Render the folium map in Streamlit
                st_folium(m, height=400, use_container_width=True)

    elif select_reg == 'All regions' and select_chart == 'Correlation Matrix':

        with col1:
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
        
        with col2:
            if len(select_columns) > 1:  # Ensure at least two columns are selected
                # Compute the correlation matrix
                correlation_matrix = df_correlation[select_columns].corr()

                # Create an interactive heatmap using Plotly without annotations or legend
                fig = px.imshow(
                    correlation_matrix,
                    color_continuous_scale='YlOrRd',
                    text_auto=".2f"
                )

                # Add titles and update layout
                fig.update_layout(
                
                            # Center the title
                width=800,              # Set width
                height=800,             # Set height
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
                coloraxis_showscale=False,  # Hide the color bar (legend)
                )

                # Render the Plotly heatmap in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Please select at least two features to compute the correlation matrix.")

    
    elif select_reg and select_chart == 'Correlation Matrix':

        if select_reg == 'Wallonia':

        
            with col1:
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
            
            with col2:
                if len(select_columns) > 1:  # Ensure at least two columns are selected
                    # Compute the correlation matrix
                    correlation_matrix = df_correlation_wal[select_columns].corr()

                    # Create an interactive heatmap using Plotly without annotations or legend
                    fig = px.imshow(
                        correlation_matrix,
                        color_continuous_scale='YlOrRd',
                        text_auto=".2f"
                    )

                    # Add titles and update layout
                    fig.update_layout(
                    
                                # Center the title
                    width=800,              # Set width
                    height=800,             # Set height
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
                    coloraxis_showscale=False,  # Hide the color bar (legend)
                    )

                    # Render the Plotly heatmap in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Please select at least two features to compute the correlation matrix.")

        if select_reg == 'Flanders':

        
            with col1:
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
            
            with col2:
                if len(select_columns) > 1:  # Ensure at least two columns are selected
                    # Compute the correlation matrix
                    correlation_matrix = df_correlation_fl[select_columns].corr()

                    # Create an interactive heatmap using Plotly without annotations or legend
                    fig = px.imshow(
                        correlation_matrix,
                        color_continuous_scale='YlOrRd',
                        text_auto=".2f"
                    )

                    # Add titles and update layout
                    fig.update_layout(
                    
                                # Center the title
                    width=800,              # Set width
                    height=800,             # Set height
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
                    coloraxis_showscale=False,  # Hide the color bar (legend)
                    )

                    # Render the Plotly heatmap in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Please select at least two features to compute the correlation matrix.")
        
        if select_reg == 'Brussels':

        
            with col1:
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
            
            with col2:
                if len(select_columns) > 1:  # Ensure at least two columns are selected
                    # Compute the correlation matrix
                    correlation_matrix = df_correlation_bx[select_columns].corr()

                    # Create an interactive heatmap using Plotly without annotations or legend
                    fig = px.imshow(
                        correlation_matrix,
                        color_continuous_scale='YlOrRd',
                        text_auto=".2f"
                    )

                    # Add titles and update layout
                    fig.update_layout(
                    
                                # Center the title
                    width=800,              # Set width
                    height=800,             # Set height
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
                    coloraxis_showscale=False,  # Hide the color bar (legend)
                    )

                    # Render the Plotly heatmap in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Please select at least two features to compute the correlation matrix.")

st.write(df_correlation)