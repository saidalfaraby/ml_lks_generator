import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# st.set_page_config(layout="wide")
st.title("SVM Interactive Companion")
st.write("Gunakan aplikasi ini sesuai instruksi pada Lembar Kerja Mahasiswa (LKS).")

# --- GENERATE DATASET BERDASARKAN NIM ---
st.header("🔑 Langkah 0: Inisialisasi Dataset")
nim_input = st.text_input("Masukkan 5 Digit Terakhir NIM Anda (hanya angka):", "")

if nim_input.isdigit():
    # Gunakan NIM sebagai seed
    seed_val = int(nim_input)
    np.random.seed(seed_val)
    
    # BRUTE-FORCE GENERATOR: Mencari dataset yang dijamin memiliki >= 3 Support Vectors
    # Sistem akan terus mengacak hingga kriteria tersebut terpenuhi
    valid_dataset_found = False
    
    while not valid_dataset_found:
        w_hidden = np.random.uniform(-1, 1, 2)
        if np.linalg.norm(w_hidden) == 0:
            continue
        w_hidden = w_hidden / np.linalg.norm(w_hidden) 
        b_hidden = np.random.uniform(-5, 5)
        
        X_neg = []
        X_pos = []
        
        # Ambil 100 sampel acak dan saring yang cukup jauh dari hidden hyperplane
        for _ in range(100):
            pt = np.random.randint(-10, 11, size=2)
            posisi = np.dot(w_hidden, pt) + b_hidden
            
            if posisi < -0.5 and len(X_neg) < 3:
                X_neg.append(pt)
            elif posisi > 0.5 and len(X_pos) < 3:
                X_pos.append(pt)
                
            if len(X_neg) == 3 and len(X_pos) == 3:
                break
                
        if len(X_neg) < 3 or len(X_pos) < 3:
            continue
            
        X_temp = np.vstack((X_neg, X_pos))
        y_temp = np.array([-1, -1, -1, 1, 1, 1])
        
        # Cepat-cepat jalankan KKT / QP Solver di balik layar untuk mengecek jumlah Support Vectors
        dot_matrix = np.dot(X_temp, X_temp.T)
        def objective(alpha):
            return 0.5 * np.sum(np.outer(alpha * y_temp, alpha * y_temp) * dot_matrix) - np.sum(alpha)
        def constraint(alpha):
            return np.dot(alpha, y_temp)
            
        cons = {'type': 'eq', 'fun': constraint}
        bounds = [(0, None) for _ in range(6)]
        
        # Eksekusi Solver
        res = minimize(objective, np.zeros(6), bounds=bounds, constraints=cons)
        
        if res.success:
            alphas = res.x
            # Hitung ada berapa titik yang menjadi Support Vector (alpha > 0)
            n_sv = np.sum(alphas > 1e-4) 
            
            # Jika Support Vectors berjumlah 3 atau lebih, kita ambil dataset ini!
            if n_sv >= 3:
                # Verifikasi keamanan ganda (pastikan data benar-benar bisa dipisah)
                w_test = np.sum((alphas * y_temp)[:, np.newaxis] * X_temp, axis=0)
                if np.linalg.norm(w_test) > 1e-3:
                    sv_indices = np.where(alphas > 1e-4)[0]
                    b_test = np.mean([y_temp[i] - np.dot(w_test, X_temp[i]) for i in sv_indices])
                    margins = y_temp * (np.dot(X_temp, w_test) + b_test)
                    
                    if np.all(margins > 0.99):
                        X = X_temp
                        y = y_temp
                        valid_dataset_found = True

    # Jika berhasil ditemukan, pisahkan kembali untuk divisualisasi
    X_neg = X[y == -1]
    X_pos = X[y == 1]
    
    # Simpan di session_state
    st.session_state['X'] = X
    st.session_state['y'] = y
    
    st.success(f"Dataset berhasil di-generate untuk NIM: {nim_input} (Dijamin minimal 3 Support Vectors)")
    
    col_data, col_plot = st.columns([1, 2])
    with col_data:
        st.write("**Catat Koordinat Ini di Kertas Anda!**")
        st.write("**Class -1 (Biru):**")
        for i, pt in enumerate(X_neg):
            st.write(f"$X_{i+1}$: ({pt[0]}, {pt[1]})")
            
        st.write("**Class +1 (Merah):**")
        for i, pt in enumerate(X_pos):
            st.write(f"$X_{i+4}$: ({pt[0]}, {pt[1]})")
            
    with col_plot:
        fig_init, ax_init = plt.subplots()
        ax_init.scatter(X_neg[:,0], X_neg[:,1], color='blue', label='Class -1', s=80)
        ax_init.scatter(X_pos[:,0], X_pos[:,1], color='red', label='Class +1', s=80)
        
        ax_init.axhline(0, color='black', linewidth=1)
        ax_init.axvline(0, color='black', linewidth=1)
        
        ax_init.set_xlim(-12, 12)
        ax_init.set_ylim(-12, 12)
        ax_init.set_xticks(np.arange(-12, 13, 1))
        ax_init.set_yticks(np.arange(-12, 13, 1))

        # Tambahkan baris ini agar rasio 1:1
        ax_init.set_aspect('equal')

        ax_init.grid(True, linestyle='--', alpha=0.6)
        ax_init.legend()
        st.pyplot(fig_init)
else:
    st.warning("Silakan masukkan NIM Anda (hanya angka) untuk men-generate data soal.")
    st.stop()

# Mengambil data dari session state
X = st.session_state['X']
y = st.session_state['y']

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Tahap 1 & 2: Eksplorasi & Jarak", "Tahap 4: Matriks & QP Solver", "Tahap 6: Verifikasi Akhir"])

# --- TAB 1: EKSPLORASI & JARAK ---
with tab1:
    st.header("Visualisasi Kandidat Hyperplane & Jarak")
    st.info("Masukkan parameter $w_1, w_2,$ dan $b$ berdasarkan hasil konversi persamaan $x_2 = a x_1 + c$ yang telah Anda kerjakan di kertas.")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parameter Hyperplane")
        w1 = st.number_input("Input w1 (koefisien x1)", value=0.0, step=0.1)
        w2 = st.number_input("Input w2 (koefisien x2)", value=1.0, step=0.1)
        b = st.number_input("Input bias (b)", value=0.0, step=0.1)
    
    # with col2:
    #     st.subheader("Titik Uji (Untuk Tahap 2)")
    #     px = st.number_input("Input x1 (Titik Data Pilihan Anda)", value=float(X[3][0]))
    #     py = st.number_input("Input x2 (Titik Data Pilihan Anda)", value=float(X[3][1]))
    with col2:
        st.subheader("Titik Uji (Tahap 2)")
        
        # Membuat daftar label informatif untuk dropdown
        point_options = []
        for i in range(len(X)):
            label_text = "-1 (Biru)" if y[i] == -1 else "+1 (Merah)"
            point_options.append(f"X{i+1} ({int(X[i][0])}, {int(X[i][1])}) -> Class {label_text}")
            
        # Menampilkan dropdown untuk memilih titik
        selected_option = st.selectbox("Pilih Titik Data dari Dataset Anda:", point_options)
        
        # Mengambil indeks dari opsi yang dipilih untuk mengekstrak koordinat x1 dan x2
        selected_idx = point_options.index(selected_option)
        px = float(X[selected_idx][0])
        py = float(X[selected_idx][1])
        
        st.info(f"**Koordinat Terpilih:** $x_1 = {int(px)}$, $x_2 = {int(py)}$")

    w = np.array([w1, w2])
    point = np.array([px, py])
    
    if st.button("Plot Garis & Hitung Jarak"):
        fig, ax = plt.subplots()
        ax.scatter(X[y==-1][:,0], X[y==-1][:,1], color='blue', label='Class -1', s=80)
        ax.scatter(X[y==1][:,0], X[y==1][:,1], color='red', label='Class +1', s=80)
        ax.scatter(px, py, color='green', marker='X', s=120, label='Titik Uji', linewidths=3)
        
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        
        xx = np.linspace(-12, 12, 100)
        if w2 != 0:
            yy = -(w1 * xx + b) / w2
            ax.plot(xx, yy, 'k-', label='Hyperplane Uji', linewidth=2)
        elif w1 != 0: 
            xx_vert = np.full_like(xx, -b/w1)
            ax.plot(xx_vert, xx, 'k-', label='Hyperplane Uji', linewidth=2)
        
        # --- TAMBAHAN KODE: PLOT VEKTOR NORMAL, RELATIF, & PROYEKSI ---
        norm_w = np.linalg.norm(w)
        if norm_w != 0:
            # 1. Plot Vektor Normal (w) dari origin (0,0)
            ax.quiver(0, 0, w1, w2, angles='xy', scale_units='xy', scale=1, 
                      color='purple', width=0.005, zorder=5, label='Vektor Normal (w)')
            ax.text(w1 * 1.1, w2 * 1.1, 'w', color='purple', fontsize=14, fontweight='bold')
            
            # 2. Menentukan Titik Referensi (R) yang memotong garis hyperplane
            if w2 != 0:
                R = np.array([0, -b/w2]) # Memotong sumbu Y
            else:
                R = np.array([-b/w1, 0]) # Memotong sumbu X (jika garis vertikal)
            
            # 3. Menghitung Vektor Relatif (v) dan Vektor Proyeksi (p)
            v = point - R
            u = w / norm_w
            p = np.dot(v, u) * u
            
            # 4. Menentukan titik jatuh proyeksi di atas garis (P_line)
            P_line = point - p
            
            # Plot Titik Referensi (R)
            ax.scatter(R[0], R[1], color='orange', marker='s', s=80, label='Titik Referensi (R)', zorder=6)
            
            # Plot Vektor Relatif (v) dari R ke Titik Uji
            ax.quiver(R[0], R[1], v[0], v[1], angles='xy', scale_units='xy', scale=1, 
                      color='orange', width=0.003, alpha=0.6, zorder=4)
            # Anotasi teks v di tengah garis relatif
            ax.text(R[0] + v[0]/2, R[1] + v[1]/2, 'v', color='orange', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            
            # Plot Vektor Proyeksi (p) dari P_line ke Titik Uji (Tegak Lurus)
            ax.quiver(P_line[0], P_line[1], p[0], p[1], angles='xy', scale_units='xy', scale=1, 
                      color='green', width=0.005, zorder=5)
            # Anotasi teks p di tengah garis proyeksi
            ax.text(P_line[0] + p[0]/2, P_line[1] + p[1]/2, 'p', color='green', fontsize=14, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            
            distance = abs(np.dot(w, point) + b) / norm_w
            st.success(f"**Jarak ortogonal (Panjang vektor p) = {distance:.4f}**")
            st.write("Gunakan visualisasi vektor **w** (ungu), **v** (oranye), dan **p** (hijau) di atas untuk memvalidasi gambar manual Anda di kertas!")
        else:
            st.error("Nilai w1 dan w2 tidak boleh keduanya nol.")
        # --------------------------------------------------------------

        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)

        # Tambahkan dua baris ini:
        ax.set_xticks(np.arange(-12, 13, 1))
        ax.set_yticks(np.arange(-12, 13, 1))

        # Tambahkan baris ini
        ax.set_aspect('equal')

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
        
        norm_w = np.linalg.norm(w)
        if norm_w != 0:
            distance = abs(np.dot(w, point) + b) / norm_w
            st.success(f"**Jarak ortogonal dari Titik ke Hyperplane = {distance:.4f}**")
            st.write("Jika garis Anda berhasil memisahkan warna merah dan biru, gunakan nilai jarak di atas untuk memvalidasi perhitungan proyeksi manual Anda di kertas!")
        else:
            st.error("Nilai w1 dan w2 tidak boleh keduanya nol.")

# --- TAB 2: MATRIKS & QP SOLVER ---
with tab2:
    st.header("Quadratic Programming Solver")
    st.write("Sesuai instruksi LKS Tahap 4, masukkan hasil perhitungan manual Matriks Dot Product ($6 \\times 6$) Anda ke dalam tabel di bawah ini.")
    st.write("Setiap sel (Baris $i$, Kolom $j$) adalah hasil dari $X_i \cdot X_j$. Perhatikan tanda minus pada koordinat Anda!")
    
    n_samples = len(y)
    
    if 'student_matrix' not in st.session_state:
        df_empty = pd.DataFrame(np.zeros((n_samples, n_samples)), 
                                columns=[f'X{i+1}' for i in range(n_samples)],
                                index=[f'X{i+1}' for i in range(n_samples)])
        st.session_state['student_matrix'] = df_empty
    
    edited_df = st.data_editor(st.session_state['student_matrix'], 
                               num_rows="fixed", 
                               use_container_width=True)
    
    if 'matrix_valid' not in st.session_state:
        st.session_state['matrix_valid'] = False

    if st.button("Validasi Matriks"):
        true_matrix = np.dot(X, X.T)
        student_matrix = edited_df.values
        
        if np.allclose(true_matrix, student_matrix, atol=0.1):
            st.success("Tepat Sekali! Susunan Matriks Dot Product Anda sudah benar.")
            st.session_state['matrix_valid'] = True
        else:
            st.error("Matriks belum tepat. Cek kembali perhitungan baris dan kolom yang masih salah.")
            st.session_state['matrix_valid'] = False
            
            for i in range(n_samples):
                for j in range(n_samples):
                    if not np.isclose(true_matrix[i, j], student_matrix[i, j], atol=0.1):
                        st.warning(f"💡 Hint: Coba periksa kembali hitungan $X_{i+1} \cdot X_{j+1}$")
                        break
                else:
                    continue
                break

    st.markdown("---")
    
    if st.session_state['matrix_valid']:
        st.info("Mesin siap menjalankan optimisasi iteratif. Silakan eksekusi Solver.")
        
        if st.button("Jalankan KKT / QP Solver"):
            def objective(alpha):
                return 0.5 * np.sum(np.outer(alpha * y, alpha * y) * edited_df.values) - np.sum(alpha)
            
            def constraint(alpha):
                return np.dot(alpha, y)
            
            cons = {'type': 'eq', 'fun': constraint}
            bounds = [(0, None) for _ in range(n_samples)]
            initial_alpha = np.zeros(n_samples)
            
            res = minimize(objective, initial_alpha, bounds=bounds, constraints=cons)
            alphas = res.x
            alphas = np.where(alphas < 1e-4, 0, alphas) 
            
            st.success("Optimisasi Selesai!")
            st.write("Nilai $\\alpha$ (Lagrange Multiplier) untuk masing-masing titik:")
            for i, a in enumerate(alphas):
                st.write(f"Titik {i+1} $X_{i+1}$: $\\alpha_{i+1}$ = **{a:.4f}**")
            
            st.warning("Catat nilai $\\alpha > 0$ di atas. Gunakan untuk melanjutkan perhitungan vektor Bobot (w) dan Bias (b) secara manual di kertas Anda.")

# --- TAB 3: VERIFIKASI AKHIR ---
with tab3:
    st.header("Verifikasi Optimal Hyperplane")
    st.write("Masukkan nilai Bobot (w) dan Bias (b) final hasil perhitungan manual Anda.")
    
    col1, col2 = st.columns(2)
    with col1:
        opt_w1 = st.number_input("Final w1", value=0.0000, step=0.0001, format="%.4f")
        opt_w2 = st.number_input("Final w2", value=0.0000, step=0.0001, format="%.4f")
    with col2:
        opt_b = st.number_input("Final bias (b)", value=0.0000, step=0.0001, format="%.4f")
        
    if st.button("Verifikasi Hyperplane (Tab 3)"):
        fig, ax = plt.subplots()
        ax.scatter(X[y==-1][:,0], X[y==-1][:,1], color='blue', label='Class -1', s=80)
        ax.scatter(X[y==1][:,0], X[y==1][:,1], color='red', label='Class +1', s=80)
        
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        
        w_opt = np.array([opt_w1, opt_w2])
        xx = np.linspace(-12, 12, 100)
        
        if opt_w2 != 0:
            # Jika w2 tidak nol (garis miring atau horizontal)
            yy = -(opt_w1 * xx + opt_b) / opt_w2
            ax.plot(xx, yy, 'k-', label='Optimal Hyperplane', linewidth=2)
            
            # Rumus margin secara aljabar: w1*x1 + w2*x2 + b = +/- 1
            yy_up = -(opt_w1 * xx + opt_b - 1) / opt_w2
            yy_down = -(opt_w1 * xx + opt_b + 1) / opt_w2
            
            ax.plot(xx, yy_up, 'k--', label='Margin +1')
            ax.plot(xx, yy_down, 'k--', label='Margin -1')
            
        elif opt_w1 != 0:
            # Jika w2 = 0 tapi w1 tidak nol (GARIS VERTIKAL)
            x_line = -opt_b / opt_w1
            ax.axvline(x_line, color='k', linestyle='-', label='Optimal Hyperplane', linewidth=2)
            
            # Garis vertikal margin
            x_margin_up = (1 - opt_b) / opt_w1
            x_margin_down = (-1 - opt_b) / opt_w1
            
            ax.axvline(x_margin_up, color='k', linestyle='--', label='Margin +1')
            ax.axvline(x_margin_down, color='k', linestyle='--', label='Margin -1')
            
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_xticks(np.arange(-12, 13, 1))
        ax.set_yticks(np.arange(-12, 13, 1))

        # Tambahkan baris ini
        ax.set_aspect('equal')

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
        
        if np.linalg.norm(w_opt) > 0:
            st.success("Jika garis margin (putus-putus) menyentuh titik-titik data terluar tanpa memotong ke dalam data, perhitungan Anda TEPAT!")
        else:
            st.error("Nilai w1 dan w2 tidak boleh keduanya nol.")
