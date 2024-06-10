import logging
import multiprocessing
import os
import yaml
import pandas as pd

from backtest import Backtest, PlanHandler
from database import Database
from fmp import FMP
from parquet import Parquet
from make_mldata import AIDataMaker
from regressor import Regressor


class MainCtx:
    def __init__(self, config):
        self.start_year = int(config['START_YEAR'])
        self.end_year = int(config['END_YEAR'])
        self.root_path = config['ROOT_PATH']
        # self.need_pq_new_year = "N"
        self.log_lvl = int(config['LOG_LVL'])
        # 다른 Class와 함수에서 connection이 자주 필요하기에 Databse Class 로 관리하지 않고 main_context로 관리
        # aws_mariadb_url = 'mysql+pymysql://' + config['MARIA_DB_USER'] + ":" + config['MARIA_DB_PASSWD'] + "@" \
        #                   + config['MARIA_DB_ADDR'] + ":" + config['MARIA_DB_PORT'] + "/" + config['MARIA_DB_NAME']
        # self.conn = sqlalchemy.create_engine(aws_mariadb_url)
        self.set_default_logger()

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

    def get_multi_logger(self):
        """
        multiprocessing용 logger를 만들어서 사용한 경우, multiprocessing이 끝난 후에 logger가 내부적으로 삭제됨
        basicConfig를 다시 세팅하여 logger를 사용하려 했으나 계속 queue is closed 에러가 발생하여 multi 로거와 default logger를
        별도로 관리함. 이 때문에 multi processor에서는 logger.info 라고, 이외에는 logging.info 라고 분리하여 작성해주어야 함(불편)
        """
        log_path = "log.txt"
        multiprocessing.freeze_support()  # for multiprocessing
        logger = logging.getLogger("multi")
        logger.setLevel(self.log_lvl)

        formatter = logging.Formatter('[%(asctime)s][%(processName)s] %(message)s (%(filename)s:%(lineno)d)')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_path, mode="a+")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def set_default_logger(self):
        log_path = "log.txt"
        # if os.path.exists(log_path):
        #     os.remove(log_path)
        logging.basicConfig(level=self.log_lvl,
                            format='[%(asctime)s][%(levelname)s][%(processName)s] '
                                   '%(message)s (%(filename)s:%(lineno)d)',
                            handlers=[logging.FileHandler(log_path, mode='a+'), logging.StreamHandler()])


def get_config():
    with open('config/conf.yaml') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def conf_check(config):
    """
    config에 온 내용에 대한 기본적인 Check Logic
    1) REPORT 종류는 EVAL, RANK, AI, AVG 뿐
    """
    for rep_type in config['REPORT_LIST']:
        if rep_type != "EVAL" and rep_type != "RANK" and rep_type != "AI" and rep_type != "AVG":
            logging.critical("Only REPORT_LIST : EVAL, RANK, AI, AVG")
            exit()


if __name__ == '__main__':
    conf = get_config()
    main_ctx = MainCtx(conf)
    conf_check(conf)

    main_ctx.create_dir("./reports")
    if conf['GET_FMP'] == "Y":
        fmp = FMP(conf, main_ctx)
        fmp.get_new()
        if conf['STORAGE_TYPE'] == "DB":
            db = Database(main_ctx)
            db.insert_csv()
            db.rebuild_table_view()
        elif conf['STORAGE_TYPE'] == "PARQUET":
            df_engine = Parquet(main_ctx)
            df_engine.insert_csv()
            df_engine.rebuild_table_view()
        else:
            logging.error("Check conf.yaml. don't choose db and parquet both")

    # TODO 잠깐 놓은거임 고쳐야댐!!!!!!!!!!!!!!!
    # df_engine = Parquet(main_ctx)
    # df_engine.insert_csv()
    # df_engine.rebuild_table_view()

    # for mem_cnt in range(5, 21, 5):
    #    for top_k_num in range(200, 3001, 400):
    #        for score_ratio in range(0, 201, 25):
    #            conf['MEMBER_CNT'] = mem_cnt
    #            conf['TOP_K_NUM'] = top_k_num
    #            conf['ABSOLUTE_SCORE'] = int(top_k_num * 10 * (1 + score_ratio / 100))


    if conf['RUN_REGRESSION'] == "Y":
        AIDataMaker(main_ctx, conf)
        regor = Regressor(conf)
        regor.dataload()
        regor.train()
        regor.evaluation()
        regor.latest_prediction()
        exit()

    
    plan_handler = PlanHandler(conf['TOP_K_NUM'], conf['ABSOLUTE_SCORE'], main_ctx)
    plan = []
    plan_df = pd.read_csv("./plan.csv")
    plan_info = plan_df.values.tolist()
    for i in range(len(plan_info)):
        plan.append(
            {"f_name": plan_handler.single_metric_plan_no_parallel,
             "params": {"key": plan_info[i][0],
                        "key_dir": plan_info[i][1], "weight": plan_info[i][2],
                        "diff": plan_info[i][3], "base": plan_info[i][4], "base_dir": plan_info[i][5]}}
        )
    plan_handler.plan_list = plan

    bt = Backtest(main_ctx, conf, plan_handler, rebalance_period=conf['REBALANCE_PERIOD'])

    del plan_handler
    del bt

    logging.shutdown()
