import streamlit as st
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

st.set_page_config(page_title="Validasi Hierarchical Clustering", layout="wide")

st.title("Aplikasi Validasi LKS Hierarchical Clustering")
st.markdown("Masukkan Matriks Jarak awal untuk melihat hasil pembentukan cluster secara otomatis.")

# Matriks Jarak Default sesuai LKS
default_matrix = np.array([
    [0.0, 9.0, 3.0, 6.0, 11.0],
    [9.0, 0.0, 7.0, 5.0, 10.0],
    [3.0, 7.0, 0.0, 9.0, 2.0],
    [6.0, 5.0, 9.0, 0.0, 8.0],
    [11.0, 10.0, 2.0, 8.0, 0.0]
])

points = ['P1', 'P2', 'P3', 'P4', 'P5']
df_default = pd.DataFrame(default_matrix, columns=points, index=points)

N = len(points)

def map_cluster_name(idx, n_base):
    idx = int(idx)
    if idx < n_base:
        return str(idx + 1)
    else:
        return chr(65 + (idx - n_base))

st.subheader("Matriks Jarak Awal")
st.dataframe(df_default.style.format("{:.2f}"))

method_map = {
    "Single Link (MIN)": "single",
    "Complete Link (MAX)": "complete",
    "Average Link (Group Average)": "average"
}

selected_method_name = st.selectbox("Pilih Metode Proximity untuk Validasi:", list(method_map.keys()))
selected_method = method_map[selected_method_name]

if st.button("Hitung & Tampilkan Dendrogram"):
    condensed_dist = squareform(default_matrix)
    Z = linkage(condensed_dist, method=selected_method)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("Langkah-langkah Penggabungan")
        df_linkage = pd.DataFrame(Z, columns=['Cluster 1', 'Cluster 2', 'Jarak Gabung', 'Anggota Baru'])
        df_linkage['Cluster 1'] = df_linkage['Cluster 1'].apply(lambda x: map_cluster_name(x, N))
        df_linkage['Cluster 2'] = df_linkage['Cluster 2'].apply(lambda x: map_cluster_name(x, N))
        df_linkage['Anggota Baru'] = df_linkage['Anggota Baru'].astype(int)
        df_linkage.index = [f"Iterasi {i+1} (Cluster {chr(65+i)})" for i in range(len(df_linkage))]
        st.dataframe(df_linkage)
        
    with col2:
        st.subheader("Dendrogram dengan Label Cluster")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simpan output dendrogram untuk mendapatkan koordinat
        icoord = []
        dcoord = []
        
        def get_dendrogram_data(linkage_matrix, labels):
            ddata = dendrogram(linkage_matrix, labels=labels, ax=ax, 
                               leaf_rotation=0, leaf_font_size=12,
                               show_leaf_counts=True)
            return ddata

        ddata = get_dendrogram_data(Z, points)

        # Logika untuk menambahkan label A, B, C, D pada internal nodes
        # icoord menyimpan posisi X garis-garis penghubung
        # dcoord menyimpan posisi Y (jarak) garis-garis penghubung
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            # Mencari indeks iterasi berdasarkan y (jarak)
            # Karena jarak mungkin sama, kita gunakan urutan dari linkage matrix Z
            for j, row in enumerate(Z):
                if np.isclose(y, row[2]):
                    cluster_label = chr(65 + j)
                    ax.plot(x, y, 'o', color='white', markersize=25, markeredgecolor=c)
                    ax.text(x, y, cluster_label, va='center', ha='center', 
                            fontweight='bold', color=c, fontsize=12)
                    break

        plt.title(f"Visualisasi {selected_method_name}", fontsize=15)
        plt.xlabel("Titik Data", fontsize=12)
        plt.ylabel("Jarak Euclidean", fontsize=12)
        st.pyplot(fig)
