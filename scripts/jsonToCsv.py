'''The json file containing the annotations was produced using MD.ai. This script
    uses the MD.ai library to convert this data into a dataframe.
    Code from https://docs.md.ai/libraries/python/guides-convert-json/ '''

import mdai

JSON = '../data/1c_mdai_rsna_project_MwBeK3Nr_annotations_labelgroup_all_2021-01-08-164102.json'
results = mdai.common_utils.json_to_dataframe(JSON)
annots_df = results['annotations']
annots_df.to_csv('../data/annotations.csv')