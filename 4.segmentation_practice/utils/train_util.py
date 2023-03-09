import logging
import os

def get_log(path, mode='train'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    foramatter = logging.Formatter('%(asctime)s - %(message)s') #포메터 설정 -> 스트림 -> 파일 저장하는 애한테 붙이고 -> 로거 
    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(foramatter)
    logger.addHandler(stream_hander)

    if mode == 'train':
        file_handler = logging.FileHandler(path + 'train.log') 
    else:
        file_handler = logging.FileHandler(path + 'eval.log') 

    file_handler.setFormatter(foramatter)
    
    logger.addHandler(file_handler)
    
    return logger
    
def create_save_dir(name):
    default_path = 'result'
    try:
        if not os.path.exists(default_path):
            os.makedirs(default_path) #make result directory
            print("create result directory !!")
            
        if not os.path.exists(default_path + '/' + name):
            os.makedirs(default_path + '/' + name)
            print('create {} directory !'.format(default_path + '/' + name))
    except:
        AssertionError('could not create dir ', + default_path + '/' + name)
        
    path = default_path + '/' + name + '/' 
    
    return path