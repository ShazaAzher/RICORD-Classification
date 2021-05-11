import pandas as pd
import os

def main():
    annotations = pd.read_csv('../data/annotations.csv')
    metadata = pd.read_csv('../data/metadata_notes.csv')
    final_annotations = []
    
    for study_id in pd.unique(annotations['StudyInstanceUID']):
        appear_label, grade_label = get_labels(annotations, study_id)
        if not appear_label: continue
        
        metadata_rows = metadata[metadata['Study UID'] == study_id]
        
        row = metadata_rows.iloc[0]
        if len(metadata_rows) > 1:
            if 'x' in list(metadata_rows['Folder']):
                row = metadata_rows[metadata_rows['Folder'] == 'x'].iloc[0]
            else:
                file_prefixes = [file.split("/")[-1][0] for file in list(metadata_rows['File Location'])]
                row = pd.Series(metadata_rows.iloc[file_prefixes.index("1")])
            
        subj_id = row['Subject ID']

        filepath = os.path.join("../data/images/manifest-1610656454899", row['File Location'][2:])
        filenames = os.listdir(filepath)
        if len(filenames) == 1:
            filepath = os.path.join(filepath, filenames[0])
        else:
            filepath = os.path.join(filepath, row['File'] + '.dcm')
        
        final_annotations.append([study_id, subj_id, filepath, appear_label, grade_label])
    
    final_annotations = pd.DataFrame(final_annotations, \
        columns=["StudyID", "SubjectID", "FilePath", "AppearLabel", "GradeLabel"])

    final_annotations.to_csv("../data/final_annotations.csv")

    
def get_labels(annotations, study_id):
    appear_counts = {'Typical Appearance': 0, 'Indeterminate Appearance': 0,
                    'Atypical Appearance': 0, 'Negative for Pneumonia': 0}
    grade_counts = {'Mild Opacities  (1-2 lung zones)': 0, 
                    'Moderate Opacities (3-4 lung zones)': 0,
                    'Severe Opacities (>4 lung zones)': 0}
    
    # all individual annotations for this study
    single_annots = annotations[annotations['StudyInstanceUID'] == study_id]

    # count appearance labels of each type
    max_appear_label = ""
    for _, row in single_annots.iterrows():
        if row['labelName'] in appear_counts:
            if row['groupName'] == 'Adjudication':
                max_appear_label = row['labelName']
                break
            else:
                appear_counts[row['labelName']] += 1
    
    # if no adjudication, find the most common appearance label
    if not max_appear_label:
        max_count = 0
        for label, count in appear_counts.items():
            if count > max_count:
                max_count = count
                max_appear_label = label
  
    
    max_grade_label = ""
    if max_appear_label != "Negative for Pneumonia": # no opacities by definition
        # count grade labels of each type
        for _, row in single_annots.iterrows():
            if row['labelName'] in grade_counts:
                if row['groupName'] == 'Adjudication':
                    max_grade_label = row['labelName']
                    break
                else:
                    grade_counts[row['labelName']] += 1
        
        # if no adjudication, find the most common grade label
        if not max_grade_label:
            max_count = 0
            for label, count in grade_counts.items():
                if count > max_count:
                    max_count = count
                    max_grade_label = label

    return max_appear_label, max_grade_label    

if __name__ == "__main__":
    main()
