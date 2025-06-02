import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import matplotlib.pyplot as plt

# --- DEFINISI VARIABEL FUZZY ---
tekanan = ctrl.Antecedent(np.arange(0.0, 6.0, 0.1), 'tekanan')
cgpa = ctrl.Antecedent(np.arange(0.0, 11.0, 0.1), 'cgpa')
jam_belajar = ctrl.Antecedent(np.arange(0.0, 13.0, 0.1), 'jam_belajar')
resiko = ctrl.Consequent(np.arange(0.0, 101.0, 0.1), 'resiko')

# --- FUZZY MEMBERSHIP FUNCTIONS ---
tekanan['rendah'] = fuzz.trimf(tekanan.universe, [0.0, 0.0, 2.0])
tekanan['sedang'] = fuzz.trimf(tekanan.universe, [1.0, 2.5, 4.0])
tekanan['tinggi'] = fuzz.trimf(tekanan.universe, [3.0, 5.0, 5.0])

cgpa['rendah'] = fuzz.trimf(cgpa.universe, [0.0, 0.0, 4.0])
cgpa['sedang'] = fuzz.trimf(cgpa.universe, [3.0, 5.0, 7.0])
cgpa['tinggi'] = fuzz.trimf(cgpa.universe, [6.0, 10.0, 10.0])

jam_belajar['rendah'] = fuzz.trimf(jam_belajar.universe, [0.0, 0.0, 4.0])
jam_belajar['sedang'] = fuzz.trimf(jam_belajar.universe, [3.0, 6.0, 9.0])
jam_belajar['tinggi'] = fuzz.trimf(jam_belajar.universe, [8.0, 12.0, 12.0])

resiko['rendah'] = fuzz.trimf(resiko.universe, [0.0, 0.0, 40.0])
resiko['sedang'] = fuzz.trimf(resiko.universe, [30.0, 50.0, 70.0])
resiko['tinggi'] = fuzz.trimf(resiko.universe, [60.0, 100.0, 100.0])

# --- FUZZY RULES ---
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

# --- READ DATASET CSV ---
df = pd.read_csv('student_depression_dataset.csv')
values = df[["id","Academic Pressure","CGPA","Work/Study Hours"]].to_numpy()
df_values = pd.DataFrame(values,columns=["id_siswa","Tekanan Akademik","CGPA","Jam Belajar"])


# --- STREAMLIT UI ---
st.title("🎓 Sistem Prediksi Risiko Depresi Mahasiswa Dengan Metode Fuzzy Logic")
st.write("##### Sistem ini merupakan aplikasi prediksi risiko depresi pada siswa/mahasiswa berbasis logika fuzzy.")

st.markdown("Silahkan Masukkan nilai berikut:")
col1,col2 =st.columns(2)
with col1:
    input_tekanan = st.slider("Tekanan Akademik (0-5)", 0, 5, 3)
    input_cgpa = st.slider("Nilai IPK / CGPA (0-10)", 0, 10, 6)
    input_jam = st.slider("Jam Belajar / Hari (0-12)", 0, 12, 5)


# --- HITUNG FUZZY ---
if st.button("🔍 Prediksi Risiko"):
    tab1,tab2 = st.tabs(["Quick Result","Detailed Result"])
    with tab1:

        #hitungan input slider
        slider_resiko_ctrl = ctrl.ControlSystem(rules)
        slider_resiko_simulasi = ctrl.ControlSystemSimulation(slider_resiko_ctrl)
        #masukkan data slider ke Fuzzy
        slider_resiko_simulasi.input['tekanan'] = input_tekanan
        slider_resiko_simulasi.input['cgpa'] = input_cgpa
        slider_resiko_simulasi.input['jam_belajar'] = input_jam

        slider_resiko_simulasi.compute()
        slider_hasil = slider_resiko_simulasi.output['resiko']

        #masukkan dataset ke fuzzy
        dataset_hasil = []
        dataset_resiko_ctrl = ctrl.ControlSystem(rules)
        for x in values:
            dataset_resiko_simulasi = ctrl.ControlSystemSimulation(dataset_resiko_ctrl)
            dataset_resiko_simulasi.input['tekanan'] = x[1]
            dataset_resiko_simulasi.input['cgpa'] = x[2]
            dataset_resiko_simulasi.input['jam_belajar'] = x[3]

            dataset_resiko_simulasi.compute()
            dataset_hasil.append(dataset_resiko_simulasi.output['resiko']) 

        # Hasil Perhitungan
        st.subheader("📈 Hasil Prediksi")
        st.write(f"Tingkat Risiko Depresi: **{slider_hasil:.2f}** dari 100")

        df_hasil = pd.DataFrame(dataset_hasil,columns=["Resiko"])
        df_gabung = pd.concat([df_values,df_hasil],axis=1)

        if slider_hasil < 40:
            st.success("Kategori Risiko: RENDAH")
            st.write("### Dataset siswa dengan Risiko Rendah : ")
            st.dataframe(df_gabung[df_gabung["Resiko"]<40])
        elif slider_hasil < 70:
            st.warning("Kategori Risiko: SEDANG")
            st.write("### Dataset siswa dengan Risiko Sedang : ")
            st.dataframe(df_gabung[(df_gabung["Resiko"]>=40) & (df_gabung["Resiko"]<70)])
        else:
            st.error("Kategori Risiko: TINGGI")
            st.write("### Dataset siswa dengan Risiko Tinggi : ")
            st.dataframe(df_gabung[df_gabung["Resiko"]>=70])
    with tab2:
        st.subheader("Detail Proses Perhitungan Fuzzy dengan Grafik")

        st.write("### 1. Tentukan input data")
        st.markdown("**Input:**")
        st.text(f"Tekanan Akademik: {input_tekanan}\nCGPA: {input_cgpa}\nJam Belajar: {input_jam}")

        st.write("### 2. Grafik Fungsi Keanggotaan Input")
        # Plot fungsi keanggotaan tekanan akademik
        fig, ax = plt.subplots(figsize=(8, 3))
        for term in tekanan.terms:
            mf = tekanan[term].mf
            ax.plot(tekanan.universe, mf, label=term)
        ax.axvline(input_tekanan, color='k', linestyle='--', label=f'Input: {input_tekanan}')
        ax.set_title("Fungsi Keanggotaan Tekanan Akademik")
        ax.legend()
        st.pyplot(fig)

        # Plot fungsi keanggotaan CGPA
        fig, ax = plt.subplots(figsize=(8, 3))
        for term in cgpa.terms:
            mf = cgpa[term].mf
            ax.plot(cgpa.universe, mf, label=term)
        ax.axvline(input_cgpa, color='k', linestyle='--', label=f'Input: {input_cgpa}')
        ax.set_title("Fungsi Keanggotaan CGPA")
        ax.legend()
        st.pyplot(fig)

        # Plot fungsi keanggotaan Jam Belajar
        fig, ax = plt.subplots(figsize=(8, 3))
        for term in jam_belajar.terms:
            mf = jam_belajar[term].mf
            ax.plot(jam_belajar.universe, mf, label=term)
        ax.axvline(input_jam, color='k', linestyle='--', label=f'Input: {input_jam}')
        ax.set_title("Fungsi Keanggotaan Jam Belajar")
        ax.legend()
        st.pyplot(fig)

        # Plot fungsi keanggotaan output resiko dengan hasil defuzzifikasi
        fig, ax = plt.subplots(figsize=(8, 3))
        for term in resiko.terms:
            mf = resiko[term].mf
            ax.plot(resiko.universe, mf, label=term)
        ax.axvline(slider_hasil, color='r', linestyle='--', label=f'Output (Resiko): {slider_hasil:.2f}')
        ax.set_title("Fungsi Keanggotaan Resiko Depresi")
        ax.legend()
        st.pyplot(fig)
        
