import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Regression Interactive LKS", layout="wide")

# ==========================================
# FUNGSI PEMBANGKIT DATA BERDASARKAN NIM
# ==========================================
def get_seed_from_nim(nim_string):
    """Mengekstrak angka dari NIM untuk dijadikan seed Numpy"""
    numbers = re.sub(r'\D', '', str(nim_string))
    if numbers:
        # Memastikan seed masuk dalam batas integer 32-bit yang diterima numpy
        return int(numbers) % (2**32 - 1)
    return 42 # Default seed jika NIM kosong/tidak valid

st.sidebar.title("Data Mahasiswa")
nim_input = st.sidebar.text_input("Masukkan NIM Anda:", "12345678")
student_seed = get_seed_from_nim(nim_input)

st.sidebar.markdown("---")
st.sidebar.title("Modul Regresi")
modul = st.sidebar.radio("Pilih Modul Pembelajaran:", 
                         ["1. Univariate Visualizer", "2. Matrix Engine (Multivariate)", "3. Logistic Explorer"])

# Main Title
st.title("Interactive Machine Learning Worksheet")
st.caption(f"Dataset dikunci menggunakan Seed NIM: {nim_input}")

# ==========================================
# MODUL 1: Univariate Visualizer
# ==========================================
if modul == "1. Univariate Visualizer":
    st.header("Modul 1: Univariate Linear Regression")
    
    # Generate Data Unique to NIM
    np.random.seed(student_seed)
    # Trik Pedagogis: X dibuat tetap dan simetris agar rata-rata X bulat (6) dan varians bulat (40)
    # Ini sangat memudahkan perhitungan manual mahasiswa di atas kertas.
    X = np.array([2, 4, 6, 8, 10]) 
    
    # Y digenerate acak berdasarkan NIM, berupa integer
    noise = np.random.randint(-15, 16, 5)
    w_true = np.random.randint(3, 8)
    a_true = np.random.randint(20, 40)
    y = w_true * X + a_true + noise
    y = np.clip(y, 10, 100) # Pastikan nilai ujian masuk akal (10-100)
    
    st.info("👇 **SALIN DATA INI KE LKS ANDA (BAGIAN 1)** 👇")
    df_univ = pd.DataFrame({"Jam Belajar (X)": X, "Nilai Ujian (Y)": y})
    # Tampilkan tabel secara horizontal agar hemat tempat
    st.dataframe(df_univ.T)
    
    st.markdown("---")
    st.write("Setelah menghitung manual, masukkan nilai bobot (w) dan bias (a) Anda di bawah ini untuk melihat garis *Best Fit* dan nilai Error.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameter Manual")
        w_input = st.number_input("Masukkan nilai bobot (w):", value=0.00, step=0.10)
        a_input = st.number_input("Masukkan nilai bias (a):", value=0.00, step=1.00)
        
        # Calculate Pred and SSE
        y_pred = w_input * X + a_input
        sse = np.sum((y - y_pred)**2)
        
        st.metric(label="Sum Square Error (SSE)", value=round(sse, 2))
        if w_input == 0 and a_input == 0:
            st.warning("Masukkan nilai w dan a hasil hitungan Anda.")
        else:
            st.success("Bandingkan nilai SSE ini. Cobalah ubah angka w dan a sedikit saja, apakah SSE membesar?")

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, y, color='blue', s=100, label='Data Aktual')
        
        # Plot garis
        x_line = np.linspace(0, 12, 100)
        y_line = w_input * x_line + a_input
        ax.plot(x_line, y_line, color='orange', linewidth=2, label=f'Model: y = {w_input:.2f}x + {a_input:.2f}')
        
        # Plot error lines (residuals)
        for i in range(len(X)):
            ax.plot([X[i], X[i]], [y[i], y_pred[i]], color='red', linestyle='--', alpha=0.5)
            
        ax.set_xlabel('Jam Belajar')
        ax.set_ylabel('Nilai Ujian')
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

# ==========================================
# MODUL 2: Matrix Engine (Multivariate)
# ==========================================
elif modul == "2. Matrix Engine (Multivariate)":
    st.header("Modul 2: Linear Regression Matrix Engine")
    
    # Generate Data Unique to NIM
    np.random.seed(student_seed + 1) # +1 agar beda dari modul 1
    X1 = np.random.choice(range(2, 10), 3, replace=False)
    X2 = np.random.randint(1, 6, 3)
    Y_multi = (3 * X1) + (4 * X2) + np.random.randint(20, 40, 3)
    
    st.info("👇 **SALIN DATA INI KE LKS ANDA (BAGIAN 2)** 👇")
    df_multi = pd.DataFrame({
        "Jam Belajar (X1)": X1, 
        "Jumlah Latihan Soal (X2)": X2, 
        "Nilai Ujian (Y)": Y_multi
    })
    st.table(df_multi)
    
    st.markdown("---")
    st.write("Susun Matriks X (jangan lupa kolom angka 1 untuk bias) dan Vektor Y di kertas Anda, lalu ketikkan ke dalam form di bawah ini.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matriks X")
        st.caption("Contoh format baris pertama: 1, 4, 2")
        X_str = st.text_area("Input Matriks X (Gunakan koma sebagai pemisah):", height=120)
        
    with col2:
        st.subheader("Vektor Y")
        st.caption("Contoh: 75")
        y_str = st.text_area("Input Vektor Y:", height=120)
        
    if st.button("Hitung Invers & Bobot Matriks", type="primary"):
        if not X_str or not y_str:
            st.error("Harap isi kedua matriks terlebih dahulu!")
        else:
            try:
                # Parse inputs
                X_mat = np.array([list(map(float, row.split(','))) for row in X_str.strip().split('\n')])
                y_vec = np.array([[float(val)] for val in y_str.strip().split('\n')])
                
                st.write("---")
                st.subheader("Langkah-langkah Komputasi Matriks:")
                
                # Step 1: X^T
                X_T = X_mat.T
                st.write("**1. Transpose Matriks ($X^T$):**")
                st.write(np.round(X_T, 4))
                
                # Step 2: X^T * X
                XTX = np.dot(X_T, X_mat)
                st.write("**2. Perkalian ($X^T X$):**")
                st.write(np.round(XTX, 4))
                
                # Step 3: Invers (X^T * X)^-1
                XTX_inv = np.linalg.inv(XTX)
                st.write("**3. Matriks Invers $(X^T X)^{-1}$:**")
                st.write(np.round(XTX_inv, 4))
                
                # Step 4: Final weights
                XTy = np.dot(X_T, y_vec)
                w = np.dot(XTX_inv, XTy)
                
                st.success("**4. Hasil Akhir Vektor Bobot (w):**")
                st.write(np.round(w, 4))
                st.info("Salin nilai w ini ke LKS Anda untuk memprediksi data testing secara manual!")
                
            except Exception as e:
                st.error("Terjadi kesalahan format input. Pastikan Anda hanya menggunakan angka dan koma, serta baris X dan Y sejajar jumlahnya.")

# ==========================================
# MODUL 3: Logistic Explorer
# ==========================================
elif modul == "3. Logistic Explorer":
    st.header("Modul 3: Logistic Regression Explorer")
    
    # Generate Data Unique to NIM
    np.random.seed(student_seed + 2)
    # Generate 10 random hours between 1 and 15
    X_log = np.sort(np.random.choice(range(1, 16), 10, replace=False))
    
    # Generate 0s and 1s with a random threshold based on NIM
    random_threshold = np.random.randint(6, 10)
    # Give some random overlaps so it's not a perfectly clean cut
    y_log = np.where(X_log + np.random.randint(-2, 3, 10) >= random_threshold, 1, 0)
    
    st.info("👇 **DATA AKTUAL ANDA (Tergambar Otomatis di Plot)** 👇")
    df_log = pd.DataFrame({"Jam Belajar (X)": X_log, "Status Lulus (Y)": y_log})
    st.dataframe(df_log.T)
    
    st.markdown("---")
    st.write("Eksplorasi secara visual bagaimana parameter $w$ dan $a$ mengubah bentuk kurva Sigmoid. Geser slider hingga **Cost (Log Loss)** sekecil mungkin.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameter Tuning")
        w_log = st.slider("Bobot (w)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
        a_log = st.slider("Bias (a)", min_value=-15.0, max_value=15.0, value=0.0, step=0.5)
        
        # Calculate probabilities and Cost
        v = w_log * X_log + a_log
        preds = 1 / (1 + np.exp(-v))
        
        # Log Loss calculation (Cost function)
        epsilon = 1e-15 # prevent log(0)
        preds_clipped = np.clip(preds, epsilon, 1 - epsilon)
        cost = -np.mean(y_log * np.log(preds_clipped) + (1 - y_log) * np.log(1 - preds_clipped))
        
        st.metric(label="Cost (Log Loss)", value=round(cost, 4))
        
        if cost < 0.4:
            st.success("Tepat Sekali! Kurva pemisah sudah cukup optimal. Salin nilai w dan a ini ke LKS Anda.")
        
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        
        # Plot scatter data
        # Warna dibedakan antara lulus dan tidak
        colors = ['red' if y == 0 else 'blue' for y in y_log]
        ax2.scatter(X_log, y_log, color=colors, s=150, edgecolor='black', zorder=5)
        
        # Plot sigmoid curve
        x_curve = np.linspace(0, 16, 200)
        v_curve = w_log * x_curve + a_log
        y_curve = 1 / (1 + np.exp(-v_curve))
        
        ax2.plot(x_curve, y_curve, color='green', linewidth=3, label=f'Sigmoid')
        
        # Threshold line
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold (0.5)')
        
        ax2.set_xlabel('Jam Belajar')
        ax2.set_ylabel('Probabilitas Lulus')
        ax2.set_xlim(0, 16)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 0.5, 1])
        ax2.set_yticklabels(['0 (Gagal)', '0.5', '1 (Lulus)'])
        ax2.legend(loc='center right')
        ax2.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig2)

# ==========================================
# FOOTER / CREDIT
# ==========================================
st.markdown("---")
st.caption("Developed by Farrel")
