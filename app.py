import streamlit as st
import pandas as pd

# -----------------------------------------
# KONFIGURASI DAN DATASET
# -----------------------------------------
st.set_page_config(page_title="Validasi LKS Naive Bayes", layout="wide")

@st.cache_data
def load_data():
    data = {
        "Battery": ["High", "Medium", "Low", "High", "Medium", "High", "Low", "Medium", "High", "Medium", "High", "Low", "Medium", "High", "Low"],
        "RAM": ["8GB", "4GB", "2GB", "8GB", "4GB", "8GB", "2GB", "4GB", "8GB", "2GB", "4GB", "2GB", "8GB", "4GB", "2GB"],
        "Storage": ["128GB", "64GB", "32GB", "128GB", "64GB", "64GB", "32GB", "128GB", "128GB", "32GB", "64GB", "64GB", "128GB", "128GB", "32GB"],
        "Camera": ["16MP", "12MP", "8MP", "16MP", "8MP", "12MP", "8MP", "12MP", "16MP", "8MP", "12MP", "8MP", "16MP", "12MP", "8MP"],
        "Price": ["Expensive", "Medium", "Cheap", "Medium", "Cheap", "Expensive", "Cheap", "Medium", "Expensive", "Cheap", "Medium", "Cheap", "Expensive", "Medium", "Cheap"],
        "Recommend": ["Yes", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "No"]
    }
    return pd.DataFrame(data)

df = load_data()
total_data = len(df)
count_yes = len(df[df['Recommend'] == 'Yes'])
count_no = len(df[df['Recommend'] == 'No'])

# -----------------------------------------
# HELPER FUNGSI VALIDASI
# -----------------------------------------
def check_fraction_or_decimal(user_input, correct_value, tolerance=0.01):
    """Memeriksa input user apakah benar (mendukung desimal dan pecahan)."""
    user_input = user_input.strip()
    if not user_input:
        return None
    try:
        if "/" in user_input:
            num, den = user_input.split("/")
            val = float(num) / float(den)
        else:
            val = float(user_input)
        return abs(val - correct_value) <= tolerance
    except ValueError:
        return False

def show_feedback(is_correct):
    if is_correct is None:
        return ""
    return "✅ Benar!" if is_correct else "❌ Kurang Tepat"

# FUNGSI CPT DENGAN LAPLACE SMOOTHING
def calculate_cpt_laplace(feature, value, target_class, k=3, alpha=1):
    subset = df[df['Recommend'] == target_class]
    count_val = len(subset[subset[feature] == value])
    total_class = len(subset)
    # Rumus Laplace: (count + alpha) / (total + k * alpha)
    return (count_val + alpha) / (total_class + (k * alpha))

# -----------------------------------------
# TAMPILAN UTAMA
# -----------------------------------------
st.title("📱 Validasi LKS: Smartphone Recommendation")
st.markdown("Aplikasi ini dibuat untuk mencocokkan hasil perhitungan mandiri algoritma **Naive Bayes**.")

with st.expander("Tampilkan Dataset Training"):
    st.dataframe(df, use_container_width=True)

# Membuat Tab untuk setiap tugas
tab1, tab2, tab3 = st.tabs(["Tugas 1: Prior Probability", "Tugas 2: CPT (Laplace)", "Tugas 3: Prediksi Data Baru"])

# --- TAB 1: PRIOR PROBABILITY ---
with tab1:
    st.header("Tugas 1 — Hitung Prior Probability")
    st.markdown("Masukkan jawaban dalam bentuk pecahan (contoh: **9/15**) atau desimal (contoh: **0.6**).")
    
    col1, col2 = st.columns(2)
    with col1:
        ans_yes = st.text_input("P(Yes) =")
        val_yes = check_fraction_or_decimal(ans_yes, count_yes / total_data)
        st.markdown(show_feedback(val_yes))
        
    with col2:
        ans_no = st.text_input("P(No) =")
        val_no = check_fraction_or_decimal(ans_no, count_no / total_data)
        st.markdown(show_feedback(val_no))

# --- TAB 2: CPT (SEKARANG DENGAN LAPLACE SMOOTHING) ---
with tab2:
    st.header("Tugas 2 — Lengkapi Conditional Probability Table (CPT) dengan Laplace Smoothing (α=1)")
    st.info("Gunakan rumus Laplace Smoothing. Input dapat berupa pecahan (misal: 7/12) atau desimal (misal: 0.58).")
    
    features = {
        "Battery": ["High", "Medium", "Low"],
        "RAM": ["8GB", "4GB", "2GB"],
        "Storage": ["128GB", "64GB", "32GB"],
        "Camera": ["16MP", "12MP", "8MP"],
        "Price": ["Expensive", "Medium", "Cheap"]
    }
    
    for feat, values in features.items():
        st.subheader(f"Feature: {feat}")
        cols = st.columns(3)
        cols[0].markdown(f"**{feat}**")
        cols[1].markdown("**P( ... | Yes )**")
        cols[2].markdown("**P( ... | No )**")
        
        for val in values:
            c = st.columns(3)
            c[0].markdown(f"*{val}*")
            
            # Input untuk Yes (Laplace)
            ans_y = c[1].text_input(f"P({val}|Yes)", key=f"{feat}_{val}_yes", label_visibility="collapsed", placeholder="misal: 7/12")
            correct_y = calculate_cpt_laplace(feat, val, "Yes")
            is_corr_y = check_fraction_or_decimal(ans_y, correct_y, tolerance=0.01)
            if is_corr_y is not None:
                c[1].markdown(show_feedback(is_corr_y))
                
            # Input untuk No (Laplace)
            ans_n = c[2].text_input(f"P({val}|No)", key=f"{feat}_{val}_no", label_visibility="collapsed", placeholder="misal: 1/9")
            correct_n = calculate_cpt_laplace(feat, val, "No")
            is_corr_n = check_fraction_or_decimal(ans_n, correct_n, tolerance=0.01)
            if is_corr_n is not None:
                c[2].markdown(show_feedback(is_corr_n))
        st.divider()

# --- TAB 3: PREDIKSI (LAPLACE SMOOTHING) ---
with tab3:
    st.header("Tugas 3 — Prediksi Data Baru dengan Laplace Smoothing (α=1)")
    st.markdown("Gunakan hasil perkalian dari nilai CPT Laplace yang sudah Anda hitung di Tugas 2.")
    
    def calc_posteriors(x_dict):
        # Prior Probabilities
        p_yes = count_yes / total_data
        p_no = count_no / total_data
        
        prob_yes = p_yes
        prob_no = p_no
        for feat, val in x_dict.items():
            prob_yes *= calculate_cpt_laplace(feat, val, "Yes")
            prob_no *= calculate_cpt_laplace(feat, val, "No")
            
        return prob_yes, prob_no

    # Tingkat toleransi ketat karena perkalian desimal menghasilkan nilai yang sangat kecil
    TOLERANCE_LAPLACE = 0.0005

    # ==========================================
    # DATA UJI 1
    # ==========================================
    st.subheader("Data Uji 1")
    st.code("X_1 = (Medium, 4GB, 64GB, 8MP, Medium)")
    
    x1_data = {"Battery": "Medium", "RAM": "4GB", "Storage": "64GB", "Camera": "8MP", "Price": "Medium"}
    true_y1, true_n1 = calc_posteriors(x1_data)
    true_pred1 = "Yes" if true_y1 > true_n1 else "No"

    st.markdown("**1. Hitung Nilai Probabilitas Akhir (Posterior)**")
    col1_1, col2_1 = st.columns(2)
    
    with col1_1:
        ans_y1 = st.text_input("Nilai P(Yes | X_1):", key="ans_y1", placeholder="Contoh: 0.0012")
        val_y1 = check_fraction_or_decimal(ans_y1, true_y1, tolerance=TOLERANCE_LAPLACE)
        st.markdown(show_feedback(val_y1))
        
    with col2_1:
        ans_n1 = st.text_input("Nilai P(No | X_1):", key="ans_n1", placeholder="Contoh: 0.0034")
        val_n1 = check_fraction_or_decimal(ans_n1, true_n1, tolerance=TOLERANCE_LAPLACE)
        st.markdown(show_feedback(val_n1))

    st.markdown("**2. Tentukan Hasil Prediksi**")
    pred1_ans = st.radio("Pilih Prediksi X_1:", ["Pilih Jawaban", "Yes", "No"], key="pred1")
    
    if pred1_ans != "Pilih Jawaban":
        if pred1_ans == true_pred1:
            st.success(f"✅ Prediksi Benar! Kelas untuk X_1 adalah {true_pred1}")
        else:
            st.error("❌ Salah. Bandingkan kembali nilai P(Yes) dan P(No) yang sudah Anda hitung.")

    st.divider()

    # ==========================================
    # DATA UJI 2
    # ==========================================
    st.subheader("Data Uji 2")
    st.code("X_2 = (Low, 4GB, 128GB, 8MP, Expensive)")
    
    x2_data = {"Battery": "Low", "RAM": "4GB", "Storage": "128GB", "Camera": "8MP", "Price": "Expensive"}
    true_y2, true_n2 = calc_posteriors(x2_data)
    true_pred2 = "Yes" if true_y2 > true_n2 else "No"

    st.markdown("**1. Hitung Nilai Probabilitas Akhir (Posterior)**")
    col1_2, col2_2 = st.columns(2)
    
    with col1_2:
        ans_y2 = st.text_input("Nilai P(Yes | X_2):", key="ans_y2", placeholder="Contoh: 0.0012")
        val_y2 = check_fraction_or_decimal(ans_y2, true_y2, tolerance=TOLERANCE_LAPLACE)
        st.markdown(show_feedback(val_y2))
        
    with col2_2:
        ans_n2 = st.text_input("Nilai P(No | X_2):", key="ans_n2", placeholder="Contoh: 0.0034")
        val_n2 = check_fraction_or_decimal(ans_n2, true_n2, tolerance=TOLERANCE_LAPLACE)
        st.markdown(show_feedback(val_n2))

    st.markdown("**2. Tentukan Hasil Prediksi**")
    pred2_ans = st.radio("Pilih Prediksi X_2:", ["Pilih Jawaban", "Yes", "No"], key="pred2")
    
    if pred2_ans != "Pilih Jawaban":
        if pred2_ans == true_pred2:
            st.success(f"✅ Prediksi Benar! Kelas untuk X_2 adalah {true_pred2}")
        else:
            st.error("❌ Salah. Bandingkan kembali nilai P(Yes) dan P(No) yang sudah Anda hitung.")
