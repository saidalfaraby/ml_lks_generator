import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Validasi LKS K-Means", layout="wide")
st.title("Validasi LKS: K-Means Clustering")
st.markdown("Masukkan hasil perhitungan manual Anda ke dalam tabel dan form di bawah ini untuk memvalidasi kebenarannya.")

# --- DATA INITIALIZATION & GROUND TRUTH (Hidden dari Mahasiswa) ---
data = {
    'ID': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8'],
    'X1': [2, 3, 2, 8, 9, 8, 1, 9],
    'X2': [3, 3, 2, 7, 8, 9, 2, 9],
    'X3': [2, 2, 1, 8, 7, 8, 1, 8],
    'X4': [1, 2, 1, 9, 8, 8, 2, 9]
}
df = pd.DataFrame(data)
X = df[['X1', 'X2', 'X3', 'X4']].values

# Ground Truth Iterasi 1
c1_init, c2_init = X[0], X[3]
true_d1, true_d2, true_cluster = [], [], []

for i in range(len(X)):
    d1 = np.sqrt(np.sum((X[i] - c1_init) ** 2))
    d2 = np.sqrt(np.sum((X[i] - c2_init) ** 2))
    true_d1.append(d1)
    true_d2.append(d2)
    true_cluster.append(1 if d1 < d2 else 2)

# Ground Truth Centroid Baru Iterasi 1
true_new_c1 = np.mean(X[np.array(true_cluster) == 1], axis=0)
true_new_c2 = np.mean(X[np.array(true_cluster) == 2], axis=0)

# Ground Truth Silhouette P1
p1_coords = X[0]
a1 = np.mean([np.sqrt(np.sum((p1_coords - X[idx]) ** 2)) for idx in [1, 2, 6]])
b1 = np.mean([np.sqrt(np.sum((p1_coords - X[idx]) ** 2)) for idx in [3, 4, 5, 7]])
true_s1 = (b1 - a1) / max(a1, b1)

# --- UI APLIKASI ---

st.header("1. Data Pelanggan")
st.dataframe(df, use_container_width=True)

st.divider()

# --- BAGIAN 1: VALIDASI TABEL JARAK ---
st.header("2. Validasi Jarak & Klaster (Iterasi 1)")
st.markdown("Isi nilai jarak (bulatkan 2 angka di belakang koma) dan pilihan klaster (1 atau 2).")

# Setup Data Editor
if 'user_df' not in st.session_state:
    st.session_state.user_df = pd.DataFrame({
        'ID': df['ID'],
        'Jarak ke C1': [0.0] * 8,
        'Jarak ke C2': [0.0] * 8,
        'Klaster Terpilih': [0] * 8
    })

edited_df = st.data_editor(
    st.session_state.user_df, 
    disabled=["ID"], 
    use_container_width=True,
    key="editor_it1"
)

if st.button("Validasi Tabel Iterasi 1", type="primary"):
    errors = []
    for i in range(8):
        user_d1 = edited_df.at[i, 'Jarak ke C1']
        user_d2 = edited_df.at[i, 'Jarak ke C2']
        user_c = edited_df.at[i, 'Klaster Terpilih']
        
        # Validasi Jarak C1 (Toleransi 0.05 untuk pembulatan)
        if not np.isclose(user_d1, true_d1[i], atol=0.02):
            errors.append(f"❌ **{df['ID'][i]}**: 'Jarak ke C1' salah. Cek kembali perhitungan Euclidean-nya.")
        
        # Validasi Jarak C2
        if not np.isclose(user_d2, true_d2[i], atol=0.02):
            errors.append(f"❌ **{df['ID'][i]}**: 'Jarak ke C2' salah.")
            
        # Validasi Penentuan Klaster
        if user_c not in [1, 2]:
            errors.append(f"❌ **{df['ID'][i]}**: 'Klaster Terpilih' harus diisi 1 atau 2.")
        elif user_c != true_cluster[i]:
            errors.append(f"❌ **{df['ID'][i]}**: 'Klaster Terpilih' salah. Ingat, pilih klaster dengan jarak terdekat!")

    if len(errors) == 0:
        st.success("🎉 Luar biasa! Semua perhitungan jarak dan penentuan klaster di Iterasi 1 sudah BENAR.")
    else:
        st.error("Masih ada perhitungan yang kurang tepat. Silakan perbaiki bagian berikut:")
        for err in errors:
            st.markdown(err)

st.divider()

# --- BAGIAN 2: VALIDASI CENTROID BARU ---
st.header("3. Validasi Centroid Baru (Setelah Iterasi 1)")
st.markdown("Masukkan koordinat rata-rata untuk centroid baru.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Centroid 1 Baru")
    c1_x1 = st.number_input("X1 (C1)", value=0.0, step=0.1)
    c1_x2 = st.number_input("X2 (C1)", value=0.0, step=0.1)
    c1_x3 = st.number_input("X3 (C1)", value=0.0, step=0.1)
    c1_x4 = st.number_input("X4 (C1)", value=0.0, step=0.1)

with col2:
    st.subheader("Centroid 2 Baru")
    c2_x1 = st.number_input("X1 (C2)", value=0.0, step=0.1)
    c2_x2 = st.number_input("X2 (C2)", value=0.0, step=0.1)
    c2_x3 = st.number_input("X3 (C2)", value=0.0, step=0.1)
    c2_x4 = st.number_input("X4 (C2)", value=0.0, step=0.1)

if st.button("Validasi Centroid Baru"):
    user_c1 = np.array([c1_x1, c1_x2, c1_x3, c1_x4])
    user_c2 = np.array([c2_x1, c2_x2, c2_x3, c2_x4])
    
    c1_correct = np.allclose(user_c1, true_new_c1, atol=0.02)
    c2_correct = np.allclose(user_c2, true_new_c2, atol=0.02)
    
    if c1_correct and c2_correct:
        st.success("🎉 Tepat sekali! Titik centroid baru Anda sudah sesuai.")
    else:
        if not c1_correct:
            st.error("❌ **Centroid 1 Baru** masih salah. Pastikan Anda hanya merata-ratakan X1, X2, X3, X4 dari anggota yang masuk ke Klaster 1.")
        if not c2_correct:
            st.error("❌ **Centroid 2 Baru** masih salah. Cek kembali rata-rata atribut anggota Klaster 2.")

st.divider()

# --- BAGIAN 3: VALIDASI SILHOUETTE ---
st.header("4. Validasi Silhouette Coefficient (Titik P1)")
st.markdown("Masukkan hasil perhitungan komponen Silhouette untuk titik P1. (Gunakan 2 angka di belakang koma)")

col_a, col_b, col_s = st.columns(3)
with col_a:
    user_a1 = st.number_input("Nilai Kohesi (a1)", value=0.00, step=0.01)
with col_b:
    user_b1 = st.number_input("Nilai Separasi (b1)", value=0.00, step=0.01)
with col_s:
    user_s1 = st.number_input("Nilai Silhouette (s1)", value=0.00, step=0.01)

if st.button("Validasi Silhouette"):
    err_sil = []
    if not np.isclose(user_a1, a1, atol=0.02):
        err_sil.append("❌ **Nilai a1 salah.** Pastikan Anda merata-ratakan jarak P1 ke P2, P3, dan P7 saja.")
    if not np.isclose(user_b1, b1, atol=0.02):
        err_sil.append("❌ **Nilai b1 salah.** Cek rata-rata jarak P1 ke P4, P5, P6, dan P8.")
    if not np.isclose(user_s1, true_s1, atol=0.02):
        err_sil.append(f"❌ **Nilai s1 salah.** Rumusnya adalah (b1 - a1) / max(a1, b1).")
        
    if len(err_sil) == 0:
        st.success("🎉 Sempurna! Pemahaman Anda tentang evaluasi klaster dengan Silhouette Coefficient sudah mantap.")
    else:
        st.error("Terdapat kesalahan pada perhitungan Silhouette:")
        for err in err_sil:
            st.markdown(err)
