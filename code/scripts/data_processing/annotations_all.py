import pandas as pd
import os

def main():
    annotations = pd.read_csv('../data/annotations.csv')
    metadata = pd.read_csv('../data/metadata_notes.csv')
    final_annotations = pd.DataFrame(columns=["StudyID", "SubjectID", "FilePath", "AppearAdjudication", 
                                    "Typical Appearance", "Indeterminate Appearance", "Atypical Appearance",
                                    "Negative for Pneumonia", "GradeAdjudication", "Mild Opacities  (1-2 lung zones)",
                                    "Moderate Opacities (3-4 lung zones)", "Severe Opacities (>4 lung zones)"])
    
    for study_id in pd.unique(annotations['StudyInstanceUID']):
        entry = {'StudyID': study_id}

        appear_counts, appear_adjudication, grade_counts, grade_adjudication = get_counts(annotations, study_id)
        if not (any(appear_counts.values()) or any(grade_counts.values())): continue

        entry['AppearAdjudication'] = appear_adjudication
        entry.update(appear_counts)
        entry['GradeAdjudication'] = grade_adjudication
        entry.update(grade_counts)
        
        metadata_rows = metadata[metadata['Study UID'] == study_id]
        
        row = metadata_rows.iloc[0]
        if len(metadata_rows) > 1:
            if 'x' in list(metadata_rows['Folder']):
                row = metadata_rows[metadata_rows['Folder'] == 'x'].iloc[0]
            else:
                file_prefixes = [file.split("/")[-1][0] for file in list(metadata_rows['File Location'])]
                row = pd.Series(metadata_rows.iloc[file_prefixes.index("1")])
            
        entry['SubjectID'] = row['Subject ID']

        filepath = os.path.join("../data/images/manifest-1610656454899", row['File Location'][2:])
        filenames = os.listdir(filepath)
        if len(filenames) == 1:
            entry['FilePath'] = os.path.join(filepath, filenames[0])
        else:
            entry['FilePath'] = os.path.join(filepath, row['File'] + '.dcm')
        
        final_annotations = final_annotations.append(entry, ignore_index=True)
    
    final_annotations.to_csv("../data/final_annotations.csv")

    
def get_counts(annotations, study_id):
    appear_counts = {'Typical Appearance': 0, 'Indeterminate Appearance': 0,
                    'Atypical Appearance': 0, 'Negative for Pneumonia': 0}
    grade_counts = {'Mild Opacities  (1-2 lung zones)': 0, 
                    'Moderate Opacities (3-4 lung zones)': 0,
                    'Severe Opacities (>4 lung zones)': 0}
    
    # all individual annotations for this study
    single_annots = annotations[annotations['StudyInstanceUID'] == study_id]

    # count appearance labels of each type
    appear_adjudication = ""
    for _, row in single_annots.iterrows():
        if row['labelName'] in appear_counts and row['groupName'] != 'Default group':
            if row['groupName'] == 'Adjudication':
                appear_adjudication = row['labelName']
            appear_counts[row['labelName']] += 1

    # count grade labels of each type  
    grade_adjudication = ""    
    for _, row in single_annots.iterrows():
        if row['labelName'] in grade_counts and row['groupName'] != 'Default group':
            if row['groupName'] == 'Adjudication':
                grade_adjudication = row['labelName']
            grade_counts[row['labelName']] += 1

    return appear_counts, appear_adjudication, grade_counts, grade_adjudication 

if __name__ == "__main__":
    main()
