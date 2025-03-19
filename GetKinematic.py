import os
import LoadD

D_folder_path = '/Users/yilinwu/Desktop/honours data/Extracted D'
D_file_path = os.path.join(D_folder_path, 'D.py')

D = LoadD.LoadD(D_file_path)	

print(D['YW-BBT01'])