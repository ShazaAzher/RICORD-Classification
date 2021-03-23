def get_label_mappings():
    appear_mapping = {}
    for line in open('../data/appear_labels.txt'):
        key, val = line.strip().split('\t')
        appear_mapping[key] = int(val)
    
    grade_mapping = {}
    for line in open('../data/grade_labels.txt'):
        key, val = line.strip().split('\t')
        grade_mapping[key] = int(val)

    return appear_mapping, grade_mapping

def 

    