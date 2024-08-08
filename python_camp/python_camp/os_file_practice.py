# --------------------------------------------
# 1. os 활용 예제 
# 
# 1) os 디렉토리 구조 출력해보기 
# 2) root directory 아래에 있는 특정 확정자 파일들 다 출력하기 
# 3) os 디렉토리 복사하기 
# --------------------------------------------
import os

def print_directory_tree(root):
    '''방법1'''
    #for elem in os.listdir(root):
    #    print(elem)

    '''방법2'''
    for elem in os.scandir(root):
        print(elem)
     
    '''방법3'''
    #with os.scandir(root) as entries:
    #    for entry in entries:
    #        print(entry.name)
     
    '''방법4'''
    # from pathlib import Path
    # entries = Path(root)
    # print(type(entries), '\n')
    # print(dir(entries), '\n')
    
    # for entry in entries.iterdir():
    #     print(entry.name)
            
def list_extension_files(root,condition):
    '''root directory 아래에 있는 특정 확정자 파일들 다 출력하기'''
    import glob
    textfiles = glob.glob(condition)
    for elem in textfiles:
        print(elem)

def copy_directory(src, des):
    '''os디렉터리 복사하기'''
    import shutil
    #shutil.copytree(src, des)는 src 경로의 폴더를 des 경로에 복사합니다.
    shutil.copytree(src,des)
    
def copy_file(src, des):
    import shutil
    #shutil.copy(src, des)
    #shutil.copyfile()
    shutil.copy(src,des)

root = r"C:\Users\user\Desktop\sesac_AI\5.python_test"
#root = os.getcwd()
#print('현재경로:',root)

#1)
#print_directory_tree(root)    

#2)
#condition = '*.py' #pickle파일은 못찾는건가?
#list_extension_files(root,condition)

#3)
#copy_directory(r'c:\Users\user\Desktop\sesac_AI\5.python_test\level_test\hello word', r'c:\Users\user\Desktop\sesac_AI\5.python_test\level_test\hello word2')
#for f in os.listdir(r'c:\Users\user\Desktop\sesac_AI\5.python_test\level_test\hello word'):
#    print(f)

#4)
#copy_file(r'C:\Users\user\Desktop\sesac_AI\5.python_test\level_test\empty_dict.pickle', r'C:\Users\user\Desktop\sesac_AI\5.python_test\level_test\empty_dict.pickle2')
