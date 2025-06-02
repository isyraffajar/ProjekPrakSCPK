import pandas as pd
import numpy as np
from collections import defaultdict

# Fungsi untuk menghitung derajat keanggotaan
def calculate_membership(x, params):
    a, b, c = params
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    return 0.0

# Fungsi untuk menentukan kategori dominan
def get_dominant_category(value, categories):
    max_degree = -1
    dominant_cat = None
    for cat, params in categories.items():
        degree = calculate_membership(value, params)
        if degree > max_degree:
            max_degree = degree
            dominant_cat = cat
    return dominant_cat

# Parameter fungsi keanggotaan (sama dengan di program utama)
membership_params = {
    'tekanan': {
        'rendah': [0, 0, 2],
        'sedang': [1, 2, 4],
        'tinggi': [3, 5, 5]
    },
    'cgpa': {
        'rendah': [0, 0, 4],
        'sedang': [3, 5, 7],
        'tinggi': [6, 10, 10]
    },
    'finansial': {
        'rendah': [0, 0, 2],
        'sedang': [1, 2, 4],
        'tinggi': [3, 5, 5]
    },
    'kepuasan': {
        'rendah': [0, 0, 2],
        'sedang': [1, 2, 4],
        'tinggi': [3, 5, 5]
    },
    'jam_belajar': {
        'rendah': [0, 0, 4],
        'sedang': [3, 6, 9],
        'tinggi': [8, 12, 12]
    }
}

# Baca dataset
df = pd.read_csv('student_depression_dataset_1.csv')

# Kelompokkan data berdasarkan kombinasi kategori
rule_groups = defaultdict(lambda: {'depressed': 0, 'total': 0})

# Proses setiap baris data
for _, row in df.iterrows():
    # Tentukan kategori dominan untuk setiap variabel
    tekanan_cat = get_dominant_category(row['Academic Pressure'], membership_params['tekanan'])
    cgpa_cat = get_dominant_category(row['CGPA'], membership_params['cgpa'])
    finansial_cat = get_dominant_category(row['Financial Stress'], membership_params['finansial'])
    kepuasan_cat = get_dominant_category(row['Study Satisfaction'], membership_params['kepuasan'])
    jam_cat = get_dominant_category(row['Work/Study Hours'], membership_params['jam_belajar'])
    
    # Buat kunci unik untuk kombinasi kategori
    rule_key = (tekanan_cat, cgpa_cat, finansial_cat, kepuasan_cat, jam_cat)
    
    # Update statistik kelompok
    rule_groups[rule_key]['total'] += 1
    if row['Depression'] == 1:
        rule_groups[rule_key]['depressed'] += 1

# Hasilkan aturan fuzzy
fuzzy_rules = []
for combination, stats in rule_groups.items():
    depression_ratio = stats['depressed'] / stats['total']
    
    # Tentukan kategori risiko berdasarkan rasio depresi
    if depression_ratio >= 0.7:
        risk_category = 'tinggi'
    elif depression_ratio <= 0.3:
        risk_category = 'rendah'
    else:
        risk_category = 'sedang'
    
    # Buat aturan fuzzy
    rule_str = (f"ctrl.Rule(tekanan['{combination[0]}'] & cgpa['{combination[1]}'] & "
                f"finansial['{combination[2]}'] & kepuasan['{combination[3]}'] & "
                f"jam_belajar['{combination[4]}'], resiko['{risk_category}'])")
    fuzzy_rules.append(rule_str)

# Simpan aturan ke file atau tampilkan
with open('generated_fuzzy_rules.py', 'w') as f:
    f.write("rules = [\n")
    for rule in fuzzy_rules:
        f.write(f"    {rule},\n")
    f.write("]\n")

print(f"Generated {len(fuzzy_rules)} fuzzy rules")