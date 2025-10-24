import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import rasterio
from folium.raster_layers import ImageOverlay
import branca.colormap as cm
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as mcm

st.set_page_config(
    page_title="Pemetaan Kriminalitas Purbalingga",
    page_icon="assets/logo.png",
    layout="wide"
)

st.markdown("""
<style>
    html, body, [class*="css"]  { font-family: 'Source Sans Pro', sans-serif; }
    .main-header { font-size: 36px; font-weight: bold; color: #1E293B; padding-bottom: 10px; }
    .sub-header { font-size: 18px; color: #475569; margin-bottom: 20px; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF; border-radius: 10px; padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); border: 1px solid #E2E8F0;
    }
    div[data-testid="stMetricLabel"] { color: #64748B; font-size: 16px; }
    .css-1d391kg { background-color: #F8FAFC; border-right: 1px solid #E2E8F0; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_all_data():
    path_prefix = 'data/'
    df_crime = pd.read_csv(path_prefix + 'kejadian.csv')
    gdf_all_kecamatan = gpd.read_file(path_prefix + 'peta_purbalingga/gadm41_IDN_3.shp')
    gdf_all_desa = gpd.read_file(path_prefix + 'peta_purbalingga/gadm41_IDN_4.shp')
    kabupaten_target = "Purbalingga"
    gdf_kec_pbg = gdf_all_kecamatan[gdf_all_kecamatan['NAME_2'] == kabupaten_target]
    gdf_desa_pbg = gdf_all_desa[gdf_all_desa['NAME_2'] == kabupaten_target]

    raster_file = 'output/peta_kerawanan_purbalingga.tif'
    with rasterio.open(raster_file) as src:
        raster_data = src.read(1, masked=True)
        raster_bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]
    
    return df_crime, gdf_kec_pbg, gdf_desa_pbg, raster_data, raster_bounds

try:
    with st.spinner('Memuat data geospasial...'):
        df_crime, gdf_kec_pbg, gdf_desa_pbg, raster_data, raster_bounds = load_all_data()
except Exception as e:
    st.error(f"❌ Gagal memuat data. Pastikan file `peta_kerawanan_purbalingga.tif` sudah dibuat. Detail: {e}")
    st.stop()

with st.sidebar:
    st.image("assets/logo.png", width=80)
    st.title("Kontrol Visualisasi")
    st.markdown("Atur layer yang ingin Anda tampilkan pada peta.")

    with st.expander(" LAYER PETA PREDIKSI", expanded=True):
        show_prediction = st.checkbox("Tampilkan Heatmap Kerawanan", True)
        opacity = st.slider("Transparansi Heatmap", 0.0, 1.0, 0.75, 0.05)
        num_classes = st.select_slider(
            "Jumlah Kelas/Band Legenda",
            options=[3, 4, 5, 6, 7],
            value=5 
        )

    with st.expander(" LAYER DATA AKTUAL", expanded=True):
        show_actual = st.checkbox("Tampilkan Titik Kejadian", True)

    with st.expander(" LAYER BATAS ADMINISTRASI", expanded=True):
        show_desa_border = st.checkbox("Tampilkan Batas Desa", False)
        show_kec_border = st.checkbox("Tampilkan Batas Kecamatan", True)

st.markdown('<p class="main-header">Dashboard Pemetaan Kriminalitas</p>', unsafe_allow_html=True)
st.markdown('<p classs="sub-header">Kabupaten Purbalingga</p>', unsafe_allow_html=True)

total_kejadian = df_crime['jumlah_kejadian'].sum()
jumlah_lokasi = len(df_crime)
rata_rata_kejadian = df_crime['jumlah_kejadian'].mean()

col1, col2, col3 = st.columns(3)
with col1: st.metric(label="Total Kejadian Kriminal", value=f"{total_kejadian} Kasus")
with col2: st.metric(label="Jumlah Lokasi Tercatat", value=f"{jumlah_lokasi} Titik")
with col3: st.metric(label="Rata-rata Kejadian per Lokasi", value=f"{rata_rata_kejadian:.1f} Kasus")
st.markdown("---")

center_lat = gdf_kec_pbg.geometry.centroid.y.mean()
center_lon = gdf_kec_pbg.geometry.centroid.x.mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None)

folium.TileLayer('https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', attr='Google Maps', name='Google Maps').add_to(m)
folium.TileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google Satellite', name='Google Satellite').add_to(m)
folium.TileLayer('CartoDB positron', name='Minimalist').add_to(m)

if show_prediction:
    min_val, max_val = np.nanmin(raster_data), np.nanmax(raster_data)
    
    cmap_colors = ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c']
    full_cmap = mcm.get_cmap('Spectral_r') 
    
    selected_colors = [colors.rgb2hex(full_cmap(i)) for i in np.linspace(0, 1, num_classes)]
    
    cmap = mcm.get_cmap(colors.ListedColormap(selected_colors))
    norm = colors.Normalize(vmin=min_val, vmax=max_val)
    colored_raster = cmap(norm(raster_data))
    colored_raster[raster_data.mask] = [0, 0, 0, 0]
    colored_raster_uint8 = (colored_raster * 255).astype(np.uint8)
    
    ImageOverlay(
        image=colored_raster_uint8,
        bounds=raster_bounds,
        opacity=opacity,
        name='Prediksi Kerawanan',
    ).add_to(m)
    steps = np.linspace(min_val, max_val, num=num_classes + 1)

    legend_colormap = cm.StepColormap(
        colors=selected_colors, 
        index=steps,            
        vmin=min_val, 
        vmax=max_val,
        caption=f"Level Kerawanan Kriminalitas ({num_classes} Kelas)"
    )
    

    m.add_child(legend_colormap)
    

if show_desa_border:
    folium.GeoJson(gdf_desa_pbg, name='Batas Desa', style_function=lambda x: {'color': "#515151", 'weight': 2, 'dashArray': '5, 5'}).add_to(m)
if show_kec_border:
    folium.GeoJson(gdf_kec_pbg, name='Batas Kecamatan', style_function=lambda x: {'color': 'black', 'weight': 2.5, 'fillOpacity': 0.0}).add_to(m)
if show_actual:
    fg_actual = folium.FeatureGroup(name='Data Aktual')
    for _, row in df_crime.iterrows():
        popup_html = f"""<div style="font-family: sans-serif; font-size: 14px;">
            <b>Lokasi Kejadian</b><hr style="margin: 2px 0;">
            <b>Jumlah:</b> {row['jumlah_kejadian']} kasus<br>
            <b>Koordinat:</b> ({row['latitude']:.4f}, {row['longitude']:.4f})</div>"""
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']], radius=6, color='#003366',
            fill=True, fill_color='#3388FF', fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=200)
        ).add_to(fg_actual)
    m.add_child(fg_actual)

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width='100%', height=600, returned_objects=[])