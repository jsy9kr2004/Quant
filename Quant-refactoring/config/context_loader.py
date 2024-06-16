import yaml
import logging
import os

class ContextLoader:
    def __init__(self):
        self.config_file = 'config/conf.yaml'
        self.log_path = 'log.txt'
        self.__load_config()

    def __load_config(self):
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
                for key, value in config.items():
                    key = key.lower()
                    if not hasattr(self, key):
                        setattr(self, key, value)
                    else:
                        raise Exception('클래스 변수 내 중복 키 존재')
        except:
            raise Exception('conf.yaml 파일 없음음')
        finally:
            logger = self.get_logger('contextLoader')
            logger.info(f'config loaded successfully')

    def get_logger(self, logger_name):
        logger = logging.getLogger(logger_name)

        # 최초 호출
        if len(logger.handlers) == 0:
            logger.setLevel('INFO')

            formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(logger_name)s] %(message)s (%(filename)s:%(lineno)d)')

            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

            file_handler = logging.FileHandler(self.log_path, mode="a+")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        extra = {'logger_name': logger_name}
        logger = logging.LoggerAdapter(logger, extra)
        
        return logger
    
    # def remove_handlers(self, logger):
    #     while len(logger.handlers) > 0:
    #         logger.removeHandler(logger.handlers[0])
    
    @staticmethod
    def create_dir(path):
        if not os.path.exists(path):
            logging.info('Creating Folder "{}" ...'.format(path))
            try:
                os.makedirs(path)
                return True
            except OSError:
                logging.error('Cannot Creating "{}" directory.'.format(path))
                return False