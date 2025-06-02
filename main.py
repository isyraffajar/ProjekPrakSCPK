import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd

# --- DEFINISI VARIABEL FUZZY ---
tekanan = ctrl.Antecedent(np.arange(0.0, 6.0, 1.0), 'tekanan')
cgpa = ctrl.Antecedent(np.arange(0.0, 11.0, 1.0), 'cgpa')
jam_belajar = ctrl.Antecedent(np.arange(0.0, 13.0, 1.0), 'jam_belajar')
resiko = ctrl.Consequent(np.arange(0.0, 101.0, 1.0), 'resiko')

# --- FUZZY MEMBERSHIP FUNCTIONS ---
tekanan['rendah'] = fuzz.trimf(tekanan.universe, [0, 0, 2])
tekanan['sedang'] = fuzz.trimf(tekanan.universe, [1, 2.5, 4])
tekanan['tinggi'] = fuzz.trimf(tekanan.universe, [3, 5, 5])

cgpa['rendah'] = fuzz.trimf(cgpa.universe, [0, 0, 4])
cgpa['sedang'] = fuzz.trimf(cgpa.universe, [3, 5.5, 7])
cgpa['tinggi'] = fuzz.trimf(cgpa.universe, [6, 10, 10])

jam_belajar['rendah'] = fuzz.trimf(jam_belajar.universe, [0, 0, 4])
jam_belajar['sedang'] = fuzz.trimf(jam_belajar.universe, [3, 6, 9])
jam_belajar['tinggi'] = fuzz.trimf(jam_belajar.universe, [8, 12, 12])

resiko['rendah'] = fuzz.trimf(resiko.universe, [0, 0, 40])
resiko['sedang'] = fuzz.trimf(resiko.universe, [30, 50, 70])
resiko['tinggi'] = fuzz.trimf(resiko.universe, [60, 100, 100])

# --- FUZZY RULES (3 VARIABEL) ---
rules = [

ctrl.Rule(tekanan['rendah'] & cgpa['rendah'] & jam_belajar['rendah'], resiko['rendah']),
ctrl.Rule(tekanan['rendah'] & cgpa['rendah'] & jam_belajar['sedang'], resiko['sedang']),
ctrl.Rule(tekanan['rendah'] & cgpa['rendah'] & jam_belajar['tinggi'], resiko['sedang']),

ctrl.Rule(tekanan['rendah'] & cgpa['sedang'] & jam_belajar['rendah'], resiko['rendah']),
ctrl.Rule(tekanan['rendah'] & cgpa['sedang'] & jam_belajar['sedang'], resiko['rendah']),
ctrl.Rule(tekanan['rendah'] & cgpa['sedang'] & jam_belajar['tinggi'], resiko['sedang']),

ctrl.Rule(tekanan['rendah'] & cgpa['tinggi'] & jam_belajar['rendah'], resiko['rendah']),
ctrl.Rule(tekanan['rendah'] & cgpa['tinggi'] & jam_belajar['sedang'], resiko['rendah']),
ctrl.Rule(tekanan['rendah'] & cgpa['tinggi'] & jam_belajar['tinggi'], resiko['sedang']),

ctrl.Rule(tekanan['sedang'] & cgpa['rendah'] & jam_belajar['rendah'], resiko['sedang']),
ctrl.Rule(tekanan['sedang'] & cgpa['rendah'] & jam_belajar['sedang'], resiko['tinggi']),
ctrl.Rule(tekanan['sedang'] & cgpa['rendah'] & jam_belajar['tinggi'], resiko['tinggi']),

ctrl.Rule(tekanan['sedang'] & cgpa['sedang'] & jam_belajar['rendah'], resiko['sedang']),
ctrl.Rule(tekanan['sedang'] & cgpa['sedang'] & jam_belajar['sedang'], resiko['sedang']),
ctrl.Rule(tekanan['sedang'] & cgpa['sedang'] & jam_belajar['tinggi'], resiko['tinggi']),

ctrl.Rule(tekanan['sedang'] & cgpa['tinggi'] & jam_belajar['rendah'], resiko['rendah']),
ctrl.Rule(tekanan['sedang'] & cgpa['tinggi'] & jam_belajar['sedang'], resiko['sedang']),
ctrl.Rule(tekanan['sedang'] & cgpa['tinggi'] & jam_belajar['tinggi'], resiko['tinggi']),

ctrl.Rule(tekanan['tinggi'] & cgpa['rendah'] & jam_belajar['rendah'], resiko['tinggi']),
ctrl.Rule(tekanan['tinggi'] & cgpa['rendah'] & jam_belajar['sedang'], resiko['tinggi']),
ctrl.Rule(tekanan['tinggi'] & cgpa['rendah'] & jam_belajar['tinggi'], resiko['tinggi']),

ctrl.Rule(tekanan['tinggi'] & cgpa['sedang'] & jam_belajar['rendah'], resiko['sedang']),
ctrl.Rule(tekanan['tinggi'] & cgpa['sedang'] & jam_belajar['sedang'], resiko['tinggi']),
ctrl.Rule(tekanan['tinggi'] & cgpa['sedang'] & jam_belajar['tinggi'], resiko['tinggi']),

ctrl.Rule(tekanan['tinggi'] & cgpa['tinggi'] & jam_belajar['rendah'], resiko['sedang']),
ctrl.Rule(tekanan['tinggi'] & cgpa['tinggi'] & jam_belajar['sedang'], resiko['sedang']),
ctrl.Rule(tekanan['tinggi'] & cgpa['tinggi'] & jam_belajar['tinggi'], resiko['tinggi'])

]
df = pd.read_csv('student_depression_dataset.csv')
# --- SISTEM KONTROL FUZZY ---
resiko_ctrl = ctrl.ControlSystem(rules)
resiko_simulasi = ctrl.ControlSystemSimulation(resiko_ctrl)

# --- STREAMLIT UI ---
st.title("🎓 Sistem Prediksi Risiko Depresi Mahasiswa Dengan Metode Fuzzy Logic")

tab1,tab2 = st.tabs
st.markdown("Masukkan nilai berikut:")
col1,col2 =st.columns(2)
with col1:
    input_tekanan = st.slider("Tekanan Akademik (0-5)", 0, 5, 3)
    input_cgpa = st.slider("Nilai IPK / CGPA (0-10)", 0, 10, 6)
    input_jam = st.slider("Jam Belajar / Hari (0-12)", 0, 12, 5)

if st.button("🔍 Prediksi Risiko"):
    resiko_simulasi.input['tekanan'] = input_tekanan
    resiko_simulasi.input['cgpa'] = input_cgpa
    resiko_simulasi.input['jam_belajar'] = input_jam

    resiko_simulasi.compute()
    hasil = resiko_simulasi.output['resiko']
    
    st.subheader("📈 Hasil Prediksi")
    st.write(f"Tingkat Risiko Depresi: **{hasil:.2f}** dari 100")

    if hasil < 40:
        st.success("Kategori Risiko: RENDAH")
    elif hasil < 70:
        st.warning("Kategori Risiko: SEDANG")
    else:
        st.error("Kategori Risiko: TINGGI")
