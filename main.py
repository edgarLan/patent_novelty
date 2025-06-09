from utils.importation import loopFinal, tar_gz2json, tar_gz2jsonDEMO
from utils.cleaning import cleanCSV
from utils.config_utils import read_config_txt
from utils.utils import createCleanTechnet, lemmatizeTechnet, measureNov, applicationPatent_match, addControlVars, createGraph, calculateCDI, metricAnalysis, output_to_excel, top10Complete



if __name__ == "__main__":
    config = read_config_txt("config.txt")
    print(config)


    # From compressed tar to json - demo only extracts first 1000 jsons
    if config["decompress"]:
        tar_gz2json(listYear=config["listYearsUnComp"], pathData=config["pathData"])
    if config["decompress_DEMO"]:
        tar_gz2jsonDEMO(listYear=config["listYearsUnComp"], pathData=config["pathData"])

    # From json to csv raw
    if config["csv_raw"]:
        loopFinal(
            listIPC=config["ipcList"],
            listYearsEval=config["yearList"],
            nbYearsRef=config["nbYearsRef"],
            pathData=config["pathData"],
            batch_size=config["batch_size"]
        )

    # From csv raw to csv clean
    ## create and clean technet
    if config["technet"]:
        createCleanTechnet(config["pathData"]) # 10 sec
        lemmatizeTechnet(config["pathData"]) # 5m30

    ## clean csvs
    if config["csv_clean"]:
        cleanCSV(config["pathData"], yearList=config["yearList"], ipcList=config["ipcList"])
        
    # From csv_clean to metrics_raw
    if config["metrics_raw"]:
        measureNov(pathData=config["pathData"], pathMetrics=config["pathMetrics"], tE_cols=config["tE_cols"], base_cols=config["base_cols"],
                    w_size=config["w_size"], yearList=config["yearList"], ipcList=config["ipcList"], chunksize=config["chunksize"], metrics_to_compute=config["metrics_to_compute"],
                    thr_new_div=config["thr_new_div"], thr_new_div_flag=config["thr_new_div_flag"], thr_new_prob=config["thr_new_prob"], thr_new_prob_flag=config["thr_new_prob_flag"],
                    thr_uniq_flag=config["thr_uniq_flag"], thr_uniqp_flag=config["thr_uniqp_flag"], useClusters=config["useClusters"], nb_clusters=config["nb_clusters"], neighbor_dist=config["neighbor_dist"],
                    thr_diff=config["thr_diff"], thr_surp=config["thr_surp"], forDemo=config["forDemo"])
        
    # Match application & patent id - might be updated to be done in json to csv raw...
    if config["match_app_pat"]:
        applicationPatent_match(listYear=config["yearList"], pathData=config["pathData"]) #2h per year

    # From metrics_raw to metrics (add control variables)
    if config["controlVars"]:
        addControlVars(ipcList=config["ipcList"], yearList=config["yearList"], sC = config["sC"], pathData=config["pathData"], pathMetrics=config["pathMetrics"]) #18s

    # Compute CD-Index
    if config["cd_index_compute"]:
        graph, valid_patent_ids = createGraph(config["pathData"]) # 18m
        calculateCDI(graph=graph, valid_patent_ids=valid_patent_ids, pathMetrics=config["pathMetrics"]) # 42m

    # Metric Analysis
    if config["metricAnalysis"]:
        df_list, sheet_names = metricAnalysis(pathMetrics=config["pathMetrics"], pathData=config["pathData"], p=config["p_rbo"], yearList=config["yearList"])
        output_to_excel(df_list=df_list, sheet_names=sheet_names, pathMetrics = config["pathMetrics"])

    if config["top10"]:
        top10Complete(pathMetrics=config["pathMetrics"], pathData=config["pathData"], ipcList=config["ipcList"], vsList=config["vsList"], yearList=config["yearList"])
