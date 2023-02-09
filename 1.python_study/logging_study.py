import logging

def main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    foramatter = logging.Formatter('%(asctime)s - %(message)s') #포메터 설정 -> 스트림 -> 파일 저장하는 애한테 붙이고 -> 로거 
    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(foramatter)
    
    logger.addHandler(stream_hander)
    file_handler = logging.FileHandler('loggingfile3.log') 
    stream_hander.setFormatter(foramatter)

    logger.addHandler(file_handler)


    #! [0]debug, [1]debug, [3]warn .. order
    logger.info("hello! Im debug")
    #logger.debug("hello,!! Im logger info")
    #logger.warning("hello!! Im logger warning")  
     
if __name__ == '__main__':
        main()
