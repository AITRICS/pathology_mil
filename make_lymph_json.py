import os, argparse, json, random
import pandas as pd
import numpy as np

def print_config(config):
    print("="*20, "Configuration", "="*20)
    print(config)
    print("="*55)

# def make_preset_data():
#     data = {}
#     data['image'] = None
#     data['DBExist'] = False
#     data['WSIExist'] = False
#     data['xmlExist'] = False
#     data['CNBDiagnosis'] = None
#     data['DiagnosisConfirmed'] = None
#     data['Annotations'] = None
#     return data

def make_meta_data(config):
    """
    Return)
        meta_data: decathlon 형식으로 image path와 label 정보가 담긴 dictionary들이 들어있는 list
    """
    DB_PATH = config.db_path
    DATA_ROOT = config.data_root
    WSI_EXTENSION_LIST = config.wsi_ext
    SAVE_PATH = config.save_path
    meta_data = []
          
    df = pd.read_excel(DB_PATH)

    for root, directories, files in os.walk(DATA_ROOT):
        for file in files:
            name, ext = os.path.splitext(file)
            # ignore Jung_DB
            if 'Jung_DB' in root: continue

            # ignore except WSI file format
            if ext not in WSI_EXTENSION_LIST: continue 

            filename_index = df.index[df['이전파일명'].str.contains(name, case=False, regex=False)]
            codipai_index = df.index[df['CODiPAI ID'].str.contains(name, case=False, regex=False)]
            db_index = None
            if len(filename_index) == 1: db_index = filename_index[0]
            if len(codipai_index) == 1: db_index = codipai_index[0]

            # ignore if name doesn't exist in 이전파일명 and CODiPAI ID columns both
            if db_index == None:
                print(f"{os.path.join(root, file)}")
                continue

            if pd.isna(df['Lymph_node_metastasis_Yes_vs_No'].iloc[db_index]): continue # ignore files don't have lymph metastasis info

            image_path = os.path.join(root, file)
            lymph_metastasis = 1 if df['Lymph_node_metastasis_Yes_vs_No'].iloc[db_index] == "Yes" else 0
            meta_data.append({'image':image_path, 'label':lymph_metastasis})
                   
    metastasis_num = 0
    not_metastasis_num = 0
    for data in meta_data:
        if data['label']:
            metastasis_num += 1
        else:
            not_metastasis_num += 1
    print(f"Total file num: {len(meta_data)}")
    print(f"metastasis: {metastasis_num}")
    print(f"not_metastasis: {not_metastasis_num}")    
    return meta_data

def split(config, meta_data):
    """
    Args)
        meta_data: decathlon 형식으로 image path와 label 정보가 담긴 dictionary들이 들어있는 list
    Return)
        meta_split_dict: label 비율에 맞추어 training과 validation으로 나뉜 dictionary
    """
    train_ratio = config.train_ratio
    pos_list = [x for x in meta_data if x['label']==1]
    neg_list = [x for x in meta_data if x['label']==0]

    train_list = []
    val_list = []

    random.shuffle(pos_list)
    random.shuffle(neg_list)
    train_list += pos_list[:int(len(pos_list)*train_ratio)]
    train_list += neg_list[:int(len(neg_list)*train_ratio)]
    val_list += pos_list[int(len(pos_list)*train_ratio):]
    val_list += neg_list[int(len(neg_list)*train_ratio):]

    meta_split_dict = {}
    meta_split_dict['training'] = train_list
    meta_split_dict['validation'] = val_list
    print(f"{len(meta_split_dict['training'])} training set")
    print(f"{len(meta_split_dict['validation'])} validation set")
    return meta_split_dict

def main(config):
    print_config(config)

    random.seed(0)
    np.random.seed(0)

    meta_data = make_meta_data(config)
    meta_split_dict = split(config, meta_data)

    with open(config.save_path, 'w') as json_meta:
        json.dump(meta_split_dict, json_meta, indent=2)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default='/workspace/pathology_mil/Thyroid_CNB_thyroid__DB_for_AITRICS_2022-06-21.xlsx')
    parser.add_argument('--data_root', type=str, default='/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK')
    parser.add_argument('--wsi_ext', type=str, nargs='+', default=['.tiff', '.svs', '.ndpi'])
    parser.add_argument('--save_path', type=str, default='/workspace/pathology_mil/lymph.json')
    parser.add_argument('--train_ratio', type=str, default=0.8)
    config = parser.parse_args()
    
    main(config)