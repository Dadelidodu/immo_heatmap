import folium
import branca
import geopandas as gpd
from streamlit_folium import folium_static
import streamlit as st

@st.cache_data
def load_data(geojson_dataset_path):
    return  gpd.read_file(geojson_dataset_path)

def load_map():
    return folium.Map(location=[50.8503, 4.3517], zoom_start=7)   

def calculate_price_range(_df):
    min_price = _df['mean_price'].min()
    max_price = _df['mean_price'].max()
    return min_price, max_price

def calculate_color_map(min_price, max_price):
    return branca.colormap.LinearColormap(['#202060',  '#00ff00'], vmin=min_price, vmax=max_price)

def style_function(feature, branca_cmap):
    style_function = {
        'fillColor': branca_cmap(feature['properties']['mean_price']),
        'color': 'black',
        'weight': 0.2,
        'fillOpacity': 0.8,}
    return style_function

def tooltip():
    tooltip = folium.GeoJsonTooltip(
            fields=['postcode', 'mean_price', 'mun_name_fr'],
            aliases=['Postal Code:', 'Price/m² (€):', 'Municipality:'],
            localize=True)
    tooltip = tooltip
    return tooltip
    
def layer(_df, branca_cmap):
    geojson_layer = folium.GeoJson(
        _df,
        style_function = lambda feature: style_function(feature, branca_cmap),
        tooltip = tooltip(),
        name="geojson_layer")
    return geojson_layer

def display_map(_df):

    # Set map
    map = load_map()
    folium.TileLayer('cartodb dark_matter').add_to(map)

    # Set variables for price range
    min_price, max_price = calculate_price_range(_df)

    # Set the color map for pricing
    branca_cmap = calculate_color_map(min_price, max_price)
    branca_cmap.add_to(map)

    # Set Geojson layer and apply styles
    geojson_layer = layer(_df, branca_cmap)
    geojson_layer.add_child(folium.GeoJsonPopup(fields=['mun_name_fr', 'postcode', 'mean_price']))
    geojson_layer.add_to(map)

    # Render the map with Streamlit's st_folium function
    return folium_static(map)