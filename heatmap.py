import geopandas as gpd
import folium
from streamlit_folium import st_folium
import streamlit as st
import os
from branca.colormap import LinearColormap
import matplotlib.colors as mcolors

# Caching the GeoJSON dataset loading
@st.cache_data
def load_geojson(filepath):
    return gpd.read_file(filepath)

# Caching the creation of the colormap
@st.cache_resource
def create_colormap(_df, cmap_name="green_red"):
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, ["lightgreen", "red"])
    min_price, max_price = df['mean_price'].min(), df['mean_price'].max()
    branca_cmap = LinearColormap(
        colors=[cmap(i / 256) for i in range(256)],  # Normalize to 0-1
        vmin=min_price,
        vmax=max_price,
        caption="Price/m² (€)",
    )
    return branca_cmap

# Caching the Folium map creation
@st.cache_resource
def create_map(_df, colormap):
    m = folium.Map(location=[50.8503, 4.3517], zoom_start=8)  # Centered at Belgium
    folium.GeoJson(
        df,
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['mean_price']),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.8,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['postcode', 'mean_price', 'mun_name_fr'],
            aliases=['Postal Code:', 'Price/m² (€):', 'Municipality:'],
            localize=True,
        ),
    ).add_to(m)
    colormap.add_to(m)
    return m

# Path to GeoJSON dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
geojson_dataset_path = os.path.join(script_dir, 'json_files/geojson_dataset.geojson')

# Load GeoJSON dataset
df = load_geojson(geojson_dataset_path)

# Create colormap
branca_cmap = create_colormap(df)

# Create a two-column layout: one for the map and one for the title/text/selection
col1, col2 = st.columns([3, 1])

with col1:
    # Create and display the folium map
    m = create_map(df, branca_cmap)
    st_folium(m, height=400)

with col2:
    st.subheader("Overview")
    st.write(
        "This heatmap shows the average property price per square meter in Belgium's various postal codes. "
        "Hover over the regions to see the average price of properties in each area."
    )
    selection = st.selectbox("Select a region", df['postcode'].unique())
    st.write(f"Selected Postcode: {selection}")