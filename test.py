from Extract_Entities import report_extrac_process,text_extrac_process,Report,ignore_conditions
import json
import time
import pandas as pd
from tqdm import tqdm
import os
import glob
import multiprocessing  # 导入多进程模块
from pathlib import Path
import configparser
from pprint import pprint
BASE_DIR = Path(__file__).resolve().parent
config_path = BASE_DIR / 'config' / 'config.ini'
conf = configparser.ConfigParser()
conf.read(config_path,encoding='utf-8')

def simple_example():
    # 示例1：包含强依存关系（应合并）和并列关系（应切分）的复杂长句
    long_sentence_1 = "肝门区可见肿块，压迫胆总管上段，胆囊未见增大，脂肪肝，请结合临床。双肺未见异常密度，支气管通畅。心脏增大，心腔密度减低，主动脉钙化。建议复查。胸廓入口水平见食道软组织结节。"
#     long_sentence_1 = """
# 腰椎退行性变,L1/2-3/4椎间盘膨出。肝门附近肝内胆管扩张。


#     """

    startTime = time.time()
    result = text_extrac_process(long_sentence_1)
    
    print("--- 示例 1 ---")
    print(f"输入长句: {long_sentence_1},耗时{time.time()-startTime:.2f}秒")
    pprint([(x['short_sentence'],x['position'],x['positive']) for x in result])

    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
def process_chunks(report_chunks):
    """处理每块Dataframe记录的函数"""
    results = []
    
    for index, row in tqdm(report_chunks.iterrows(), total=len(report_chunks), desc='子进程处理进度：'):
        # 替换DX为DR
        modality = row.get('类型', '')
        if pd.notna(modality):
            modality = modality.replace("DX", "DR")
        else:
            continue
        
        # 跳过不符合条件的记录
        if modality not in ["CT", "MR", "DR"]:
            continue
        if not isinstance(row['描述'], str) or not isinstance(row['结论'], str) or not isinstance(row['部位'], str):
            continue
        # 检查是否符合忽略条件
        study_part = row.get('部位', '')
        if any(condition.get('modality') == modality and condition.get('部位') in str(study_part) 
               for condition in ignore_conditions):
            continue

        report = Report(
            ReportStr=row['描述'],
            ConclusionStr=row['结论'],
            StudyPart = row['部位'],
            Sex =  str(row['性别']),
            modality = modality,
            applyTable=str(row['申请单'])
        )
        record= report_extrac_process(report)
        record = [{**item, 'Accno': row['影像号']} for item in record]
        if record :
            results.extend(record)
    # 创建DataFrame
    return pd.DataFrame(results)

def process_corpus_excel_files(corpus_dir='corpus', num_processes=12):
    """
    读取corpus目录下所有的excel并合并。将每条记录的"描述"和"结论"字段按照标点符号分句，
    然后合并成一个list。最后输出一个两列的dataframe，第一列是分句后的单个句子，
    第二列是原本这行对应的"部位"字段。
    
    Args:
        corpus_dir (str): 包含Excel文件的目录路径，默认为'corpus'
        num_processes (int): 进程池中的进程数量，默认为6
        
    Returns:
        pd.DataFrame: 包含两列的DataFrame，第一列为句子，第二列为部位
    """
    # 获取所有Excel文件路径
    excel_files = glob.glob(os.path.join(corpus_dir, "*.xlsx"))
    
    # 读取并合并所有Excel文件
    dataframes = []
    for file in excel_files:
        try:
            df = pd.read_excel(file)
            dataframes.append(df)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
    
    # 合并所有数据
    if not dataframes:
        raise ValueError("未找到任何有效的Excel文件")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"完成合并{len(combined_df)}份报告")
    # 将报告库分割成num_processes个部分
    startTime=time.time()
    report_chunks = [combined_df[i::num_processes] for i in range(num_processes)]
    
    # 使用进程池并行处理

    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度
        sentences_with_parts=list(tqdm(pool.imap_unordered(process_chunks, report_chunks), total=len(report_chunks), desc="句子处理中..."))
    
    # 创建结果DataFrame
    result_df = pd.concat(sentences_with_parts, ignore_index=True) 
    # result_df=process_chunks(combined_df) 
    print(f"处理完成,获得{len(result_df)}行数据")
    

    # 去重
    result_df = result_df.drop_duplicates("short_sentence").reset_index(drop=True)
    print(f"去重后,获得{len(result_df)}行数据,耗时{time.time()-startTime:.2f}秒,平均速度{(time.time()-startTime)/len(combined_df)*1000:.4f}毫秒/份")
    
    return result_df

if __name__ == '__main__':
    # simple_example()
    result_df=process_corpus_excel_files()
    # # 将结果保存到Excel文件
    output_file = BASE_DIR / 'processed_copus' / 'processed_report_data'
    (result_df[:int(len(result_df)/2)]).to_excel(str(output_file) + "1.xlsx", index=False)
    (result_df[int(len(result_df)/2):]).to_excel(str(output_file) + "2.xlsx", index=False)
