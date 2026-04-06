import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Real RNN & LSTM Simulation")
st.title("Simulasi Real RNN & LSTM (NumPy Implementation)")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

tab1, tab2, tab3 = st.tabs([
    "1. BPTT & Aktivasi (RNN)", 
    "2. Perbandingan Memori: RNN vs LSTM",
    "3. Solusi Vanishing Gradient (Apple-to-Apple)"
])

# ==========================================
# TAB 1: REAL RNN (DENGAN PILIHAN AKTIVASI)
# ==========================================
with tab1:
    st.header("Dampak Fungsi Aktivasi pada Aliran Gradien (BPTT)")
    st.markdown("Pilih fungsi aktivasi untuk melihat bagaimana turunannya (*derivative*) mempengaruhi gradien saat proses *backward pass*.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Hyperparameter RNN")
        w_hh = st.slider("Bobot Hidden (W_hh)", min_value=0.1, max_value=2.0, value=1.2, step=0.1, key='w_hh_tab1')
        activation = st.selectbox("Fungsi Aktivasi", ["Tanh", "Linear", "ReLU", "Sigmoid"])
        w_xh = 0.5 
        t_steps = 50
        
    with col2:
        x = np.random.uniform(-0.1, 0.1, t_steps)
        h = np.zeros(t_steps)
        z = np.zeros(t_steps) 
        
        for t in range(1, t_steps):
            z[t] = w_hh * h[t-1] + w_xh * x[t]
            if activation == "Linear": h[t] = z[t]
            elif activation == "Tanh": h[t] = np.tanh(z[t])
            elif activation == "ReLU": h[t] = relu(z[t])
            elif activation == "Sigmoid": h[t] = sigmoid(z[t])
            
        dh = np.zeros(t_steps)
        dh[-1] = 1.0  
        
        for t in reversed(range(t_steps - 1)):
            if activation == "Linear": deriv = 1.0
            elif activation == "Tanh": deriv = 1.0 - h[t+1]**2
            elif activation == "ReLU": deriv = 1.0 if z[t+1] > 0 else 0.0
            elif activation == "Sigmoid": deriv = h[t+1] * (1.0 - h[t+1])
            dh[t] = dh[t+1] * w_hh * deriv
            
        df_grad = pd.DataFrame({"Time Step": range(t_steps), "Besaran Gradien (dh)": dh})
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_grad["Time Step"], df_grad["Besaran Gradien (dh)"], color='red', marker='.')
        ax.set_yscale('log')
        ax.set_xlabel("Time Step (0 = Awal Sekuens, 50 = Akhir Sekuens)")
        ax.set_ylabel("Gradien dL/dh (Log Scale)")
        ax.set_title(f"Aliran Gradien Mundur (Aktivasi: {activation}, W_hh = {w_hh})")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ==========================================
# TAB 2: PERBANDINGAN RNN vs LSTM
# ==========================================
with tab2:
    st.header("Pertarungan Memori: Vanilla RNN vs LSTM")
    st.markdown("Kalimat Uji: **'Film ini TIDAK bagus sama sekali bro'**. Kata **'TIDAK'** ($x_t = 2.0$) adalah informasi krusial.")
    
    col_gate, col_plot = st.columns([1, 2.5])
    
    with col_gate:
        st.subheader("Intervensi Bias LSTM")
        b_f = st.slider("Bias Forget Gate (b_f)", -5.0, 5.0, 1.0, step=0.5, key='bf_tab2')
        b_i = st.slider("Bias Input Gate (b_i)", -5.0, 5.0, -1.0, step=0.5, key='bi_tab2')
        b_o = st.slider("Bias Output Gate (b_o)", -5.0, 5.0, 0.0, step=0.5, key='bo_tab2')

    with col_plot:
        words = ["Film", "ini", "TIDAK", "bagus", "sama", "sekali", "bro"]
        x_seq = np.array([0.1, 0.1, 2.0, 0.1, 0.1, 0.1, 0.1]) 
        
        W, U, b_rnn = 0.5, 0.5, 0.0
        seq_len = len(words)
        
        h_rnn_list = np.zeros(seq_len)
        C_lstm_list = np.zeros(seq_len)
        h_lstm_list = np.zeros(seq_len)
        
        h_prev_rnn, c_prev_lstm, h_prev_lstm = 0.0, 0.0, 0.0
        
        for t in range(seq_len):
            h_rnn = np.tanh(W * x_seq[t] + U * h_prev_rnn + b_rnn)
            h_rnn_list[t] = h_rnn
            h_prev_rnn = h_rnn
            
            f_t = sigmoid(W * x_seq[t] + U * h_prev_lstm + b_f)
            i_t = sigmoid(W * x_seq[t] + U * h_prev_lstm + b_i)
            o_t = sigmoid(W * x_seq[t] + U * h_prev_lstm + b_o)
            
            c_cand = np.tanh(W * x_seq[t] + U * h_prev_lstm + 0.0)
            c_t = (f_t * c_prev_lstm) + (i_t * c_cand)
            h_lstm = o_t * np.tanh(c_t)
            
            C_lstm_list[t] = c_t
            h_lstm_list[t] = h_lstm
            c_prev_lstm, h_prev_lstm = c_t, h_lstm
            
        df_compare = pd.DataFrame({
            "Time Step (t)": range(1, seq_len + 1), "Kata": words, "Input (x_t)": x_seq,
            "RNN: Memori (h_t)": h_rnn_list, "LSTM: Cell State (C_t)": C_lstm_list, "LSTM: Output (h_t)": h_lstm_list
        })
        st.dataframe(df_compare.style.format("{:.3f}", subset=["Input (x_t)", "RNN: Memori (h_t)", "LSTM: Cell State (C_t)", "LSTM: Output (h_t)"]).background_gradient(cmap='Oranges', subset=["RNN: Memori (h_t)"]).background_gradient(cmap='Blues', subset=["LSTM: Cell State (C_t)"]), height=300)

# ==========================================
# TAB 3: BPTT LSTM (APPLE-TO-APPLE)
# ==========================================
with tab3:
    st.header("Apple-to-Apple: Bagaimana Gate LSTM Menyelamatkan Gradien")
    st.markdown("""
    Kita gunakan **Bobot Recurrent (W)** yang sama persis untuk Vanilla RNN maupun fungsi aktivasi gerbang di LSTM. 
    Mari kita buktikan bahwa dalam kondisi seburuk apa pun (W sangat kecil / sangat besar), LSTM bisa terselamatkan hanya dengan mengubah Bias Gate-nya.
    """)

    col_bptt_ctrl, col_bptt_plot = st.columns([1, 2.5])

    with col_bptt_ctrl:
        st.subheader("Parameter Bersama")
        w_shared = st.slider("Bobot Recurrent (W)", 0.1, 1.5, 0.8, 0.1, key='w_shared_tab3',
                             help="Digunakan sebagai W_hh di RNN dan sebagai bobot recurrent pembentuk f_t di LSTM. Ini menjamin perbandingan yang adil.")
        
        st.markdown("---")
        st.subheader("Intervensi Khusus LSTM")
        b_f_lstm = st.slider("Bias Forget Gate (b_f)", -3.0, 3.0, 1.0, 0.5, key='bf_tab3')
        
        st.markdown("---")
        st.markdown("### 💡 Lesson Learned:")
        if b_f_lstm > 0:
            st.success(f"**Jalan Tol Gradien (Highway):** Meskipun bobot W={w_shared}, garis merah (RNN) hancur karena aktivasi Tanh yang berulang. Namun, garis biru (LSTM) tetap stabil! Kenapa? Karena Bias Forget positif ({b_f_lstm}) menjaga nilai $f_t \\approx 1$. Gradien mengalir bebas lewat 'jalan tol' Cell State.")
        else:
            st.error("**Amnesia Buatan:** Saat Bias Forget negatif, Anda memaksa $f_t$ mendekati 0. LSTM kehilangan keajaibannya dan gradiennya (garis biru) ikut lenyap ditelan bumi bersama RNN. Ini membuktikan bahwa arsitektur Gate adalah kuncinya.")

    with col_bptt_plot:
        t_steps = 50
        np.random.seed(42) 
        x_rand = np.random.uniform(-0.1, 0.1, t_steps)
        
        # 1. Hitung BPTT RNN (Menggunakan w_shared)
        h_rnn = np.zeros(t_steps)
        for t in range(1, t_steps):
            h_rnn[t] = np.tanh(w_shared * h_rnn[t-1] + 0.5 * x_rand[t])
            
        dh_rnn = np.zeros(t_steps)
        dh_rnn[-1] = 1.0
        for t in reversed(range(t_steps - 1)):
            dh_rnn[t] = dh_rnn[t+1] * w_shared * (1 - h_rnn[t+1]**2)
            
        # 2. Hitung BPTT LSTM Cell State (Menggunakan w_shared)
        c_lstm = np.zeros(t_steps)
        h_lstm_dummy = np.zeros(t_steps)
        f_gates = np.zeros(t_steps)
        
        for t in range(1, t_steps):
            # Menggunakan w_shared untuk mengontrol seberapa besar pengaruh h sebelumnya terhadap Forget Gate
            f_gates[t] = sigmoid(0.5 * x_rand[t] + w_shared * h_lstm_dummy[t-1] + b_f_lstm)
            h_lstm_dummy[t] = 0.0 # Simplify h flow untuk mengisolasi efek murni dari Gate
            
        dc_lstm = np.zeros(t_steps)
        dc_lstm[-1] = 1.0
        for t in reversed(range(t_steps - 1)):
            dc_lstm[t] = dc_lstm[t+1] * f_gates[t+1]

        # 3. Plotting Perbandingan
        df_bptt = pd.DataFrame({
            "Time Step": range(t_steps),
            "Gradien RNN (dh)": dh_rnn,
            "Gradien LSTM (dC)": dc_lstm
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_bptt["Time Step"], df_bptt["Gradien RNN (dh)"], color='red', marker='.', linestyle='--', label=f'Vanilla RNN (W = {w_shared})')
        ax.plot(df_bptt["Time Step"], df_bptt["Gradien LSTM (dC)"], color='blue', marker='o', linewidth=2, label=f'LSTM (W = {w_shared}, b_f = {b_f_lstm})')
        
        ax.set_yscale('log')
        ax.axhline(1.0, color='green', linestyle=':', alpha=0.5, label='Batas Stabil (Ideal)')
        ax.set_xlabel("Time Step (Mundur dari Kanan ke Kiri)")
        ax.set_ylabel("Besaran Gradien (Log Scale)")
        ax.set_title("Apple-to-Apple BPTT: Mengapa Gate LSTM Menang")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
