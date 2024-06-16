from config.context_loader import ContextLoader
from data_collector.fmp import FMP

def main():
    main_ctx = ContextLoader()
    logger = main_ctx.get_logger('main')

    main_ctx.create_dir("./reports")
    if main_ctx.get_fmp == "Y":
        fmp = FMP(main_ctx)
        fmp.collect()
        # if main_ctx['STORAGE_TYPE'] == "DB":
        #     db = Database(main_ctx)
        #     db.insert_csv()
        #     db.rebuild_table_view()
        # elif main_ctx['STORAGE_TYPE'] == "PARQUET":
        #     df_engine = Parquet(main_ctx)
        #     df_engine.insert_csv()
        #     df_engine.rebuild_table_view()
        # else:
        #     logger.error("Check conf.yaml. don't choose db and parquet both")


if __name__ == '__main__':
    main()