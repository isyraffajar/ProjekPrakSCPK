import streamlit as st
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- DEFINISI VARIABEL ---
tekanan = ctrl.Antecedent(np.arange(0.0, 6.0, 1.0), 'tekanan')
cgpa = ctrl.Antecedent(np.arange(0.0, 11.0, 1.0), 'cgpa')
finansial = ctrl.Antecedent(np.arange(0.0, 6.0, 1.0), 'finansial')
kepuasan = ctrl.Antecedent(np.arange(0.0, 6.0, 1.0), 'kepuasan')
jam_belajar = ctrl.Antecedent(np.arange(0.0, 13.0, 1.0), 'jam_belajar')
resiko = ctrl.Consequent(np.arange(0.0, 101.0, 1.0), 'resiko')

# --- FUZZY FUNGSI KEANGGOTAAN ---
tekanan['rendah'] = fuzz.trimf(tekanan.universe, [0, 0, 2])
tekanan['sedang'] = fuzz.trimf(tekanan.universe, [1, 2, 4])
tekanan['tinggi'] = fuzz.trimf(tekanan.universe, [3, 5, 5])

cgpa['rendah'] = fuzz.trimf(cgpa.universe, [0, 0, 4])
cgpa['sedang'] = fuzz.trimf(cgpa.universe, [3, 5, 7])
cgpa['tinggi'] = fuzz.trimf(cgpa.universe, [6, 10, 10])

finansial['rendah'] = fuzz.trimf(finansial.universe, [0, 0, 2])
finansial['sedang'] = fuzz.trimf(finansial.universe, [1, 2, 4])
finansial['tinggi'] = fuzz.trimf(finansial.universe, [3, 5, 5])

kepuasan['rendah'] = fuzz.trimf(kepuasan.universe, [0, 0, 2])
kepuasan['sedang'] = fuzz.trimf(kepuasan.universe, [1, 2, 4])
kepuasan['tinggi'] = fuzz.trimf(kepuasan.universe, [3, 5, 5])

jam_belajar['rendah'] = fuzz.trimf(jam_belajar.universe, [0, 0, 4])
jam_belajar['sedang'] = fuzz.trimf(jam_belajar.universe, [3, 6, 9])
jam_belajar['tinggi'] = fuzz.trimf(jam_belajar.universe, [8, 12, 12])

resiko['rendah'] = fuzz.trimf(resiko.universe, [0, 0, 40])
resiko['sedang'] = fuzz.trimf(resiko.universe, [30, 50, 70])
resiko['tinggi'] = fuzz.trimf(resiko.universe, [60, 100, 100])

# --- FUZZY RULES ---
rules = [
    ctrl.Rule(tekanan['tinggi'] & cgpa['rendah'] & finansial['tinggi'] & kepuasan['rendah'] & jam_belajar['rendah'], resiko['tinggi']),
    ctrl.Rule(tekanan['tinggi'] & cgpa['sedang'] & finansial['tinggi'] & kepuasan['sedang'] & jam_belajar['rendah'], resiko['tinggi']),
    ctrl.Rule(tekanan['rendah'] & cgpa['tinggi'] & finansial['rendah'] & kepuasan['tinggi'] & jam_belajar['tinggi'], resiko['rendah']),
    ctrl.Rule(tekanan['sedang'] & cgpa['sedang'] & finansial['sedang'] & kepuasan['sedang'] & jam_belajar['sedang'], resiko['sedang']),
    ctrl.Rule(tekanan['rendah'] & cgpa['tinggi'] & finansial['sedang'] & kepuasan['tinggi'] & jam_belajar['tinggi'], resiko['rendah']),
    ctrl.Rule(tekanan['tinggi'] & cgpa['rendah'] & finansial['sedang'] & kepuasan['rendah'] & jam_belajar['sedang'], resiko['tinggi']),
    ctrl.Rule(tekanan['rendah'] & cgpa['sedang'] & finansial['tinggi'] & kepuasan['sedang'] & jam_belajar['sedang'], resiko['sedang']),
    ctrl.Rule(tekanan['sedang'] & cgpa['tinggi'] & finansial['tinggi'] & kepuasan['tinggi'] & jam_belajar['rendah'], resiko['sedang']),
    ctrl.Rule(tekanan['sedang'] & cgpa['rendah'] & finansial['sedang'] & kepuasan['tinggi'] & jam_belajar['tinggi'], resiko['sedang']),
    ctrl.Rule(tekanan['tinggi'] & cgpa['sedang'] & finansial['tinggi'] & kepuasan['tinggi'] & jam_belajar['tinggi'], resiko['tinggi']),
    
]

resiko_ctrl = ctrl.ControlSystem(rules) #membuat sistem fuzzy dengan rules yang sudah dibuat

#masukkan dataset
df = pd.read_csv('student_depression_dataset.csv')
df_value = df[["Academic Pressure","CGPA","Financial Stress","Study Satisfaction","Work/Study Hours"]].to_numpy()



# --- STREAMLIT UI ---
st.title("🎓 Fuzzy Sistem Prediksi Risiko Depresi Siswa")
st.markdown("Masukkan nilai untuk masing-masing kriteria:")

input_tekanan = st.slider("Tekanan Akademik (0-5)", 0, 5, 3)
input_cgpa = st.slider("Nilai IPK/CGPA (0-10)", 0, 10, 6)
input_finansial = st.slider("Beban Finansial (0-5)", 0, 5, 2)
input_kepuasan = st.slider("Kepuasan Belajar (0-5)", 0, 5, 3)
input_jam = st.slider("Jam Belajar per Hari (0-12)", 0, 12, 5)

if st.button("🔍 Hitung Risiko"):
    # --- Simulasi Perhitungan Fuzzy dengan Slider  ---
    resiko_simulasi = ctrl.ControlSystemSimulation(resiko_ctrl)
    resiko_simulasi.input['tekanan'] = input_tekanan
    resiko_simulasi.input['cgpa'] = input_cgpa
    resiko_simulasi.input['finansial'] = input_finansial
    resiko_simulasi.input['kepuasan'] = input_kepuasan
    resiko_simulasi.input['jam_belajar'] = input_jam

    resiko_simulasi.compute()
    hasil = resiko_simulasi.output['resiko']

    # --- Simulasi Perhitungan Fuzzy dengan Dataset  ---
    daset_hasil = []
    for x in df_value:
        daset_resiko_simulasi = ctrl.ControlSystemSimulation(resiko_ctrl)
        daset_resiko_simulasi.input['tekanan'] = x[0]
        daset_resiko_simulasi.input['cgpa'] = x[1]
        daset_resiko_simulasi.input['finansial'] = x[2]
        daset_resiko_simulasi.input['kepuasan'] = x[3]
        daset_resiko_simulasi.input['jam_belajar'] = x[4]

        daset_resiko_simulasi.compute()
        st.write(f"{daset_resiko_simulasi.output}")
        daset_hasil.append(daset_resiko_simulasi.output['resiko']) 

    # Tampilkan hasil
    st.subheader("📈 Hasil Prediksi")
    st.write(f"Tingkat Risiko Depresi: **{hasil:.2f}** dari 100")

    if hasil < 40:
        st.success("Kategori Risiko: RENDAH")
    elif hasil < 70:
        st.warning("Kategori Risiko: SEDANG")
    else:
        st.error("Kategori Risiko: TINGGI")
