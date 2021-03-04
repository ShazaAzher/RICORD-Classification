import pandas as pd

annotations = pd.read_csv('../data/annotations.csv')
metadata = pd.read_csv('../data/metadata_notes.csv')
labels = {}

for study_id in pd.unique(annotations['StudyInstanceUID']):
    appear_label, grade_label = get_labels(study_id)
    
# def get_labels(study_id):
for index, row in annotations.iterrows():
    if row['removed'] == "R": continue
    
    if row['StudyInstanceUID'] in labels:
        labels[row['StudyInstanceUID']].append((row['groupName'], row['labelName']))
    else:
        labels[row['StudyInstanceUID']] = [(row['groupName'], row['labelName'])]

for study_id in labels:
    appear_counts = {'Typical Appearance': 0, 'Indeterminate Appearance': 0,
                    'Atypical Appearance': 0, 'Negative for Pneumonia': 0}
    grade_counts = {'Mild Opacities (1-2 lung zones)': 0, 
                    'Moderate Opacities (3-4 lung zones)': 0,
                    'Severe Opacities (>4 lung zones)': 0}
    
    # count appearance labels of each type
    max_appear_label = ""
    for anno in labels[study_id]:
        if anno[1] in appear_counts and anno[0] == 'Adjudication':
            max_appear_label = anno[1]
            break
        if anno[1] in appear_counts:
            appear_counts[anno[1]] += 1
    
    # if no adjudication, find the most common appearance label
    if not max_appear_label:
        max_count = -1
        for label, count in appear_counts.items():
            if count > max_count:
                max_count = count
                max_appear_label = label
  
    
    max_grade_label = ""
    if max_appear_label != "Negative for Pneumonia": # no opacities by definition
        # count grade labels of each type
        for anno in labels[study_id]:
            if anno[1] in grade_counts and anno[0] == 'Adjudication':
                max_grade_label = anno[1]
                break
            if anno[1] in grade_counts:
                grade_counts[anno[1]] += 1
        
        # if no adjudication, find the most common grade label
        if not max_grade_label:
            max_count = -1
            for label, count in grade_counts.items():
                if count > max_count:
                    max_count = count
                    max_grade_label = label


