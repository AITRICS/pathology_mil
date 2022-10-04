import os
import glob

ext_from = '.ndpi'
ext_to = '.xml'

l = glob.glob(f'/nfs/thena/shared/Thyroid_Needle_Biopsy_JungCK/AITRICS_Core_Needle_Biopsy_annotated_SS_THCB_1959-2168(210)/*{ext_from}')

for f in l:
    f_ = f.replace(ext_from, ext_to)
    if not os.path.exists(f_):
        print(f'{f_}')
print(f'count: {len(l)}')
