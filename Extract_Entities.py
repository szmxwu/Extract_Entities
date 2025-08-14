from flashtext.keyword import KeywordProcessor  # 源码被修改过，文件为keyword.py
import pandas as pd
from collections import defaultdict
import numpy as np
import time
import re
from tqdm import tqdm  # 进度条库
from typing import List, Dict, Tuple,Any
import configparser
from tools.disambiguation import disambiguate_entities
from functools import lru_cache
from pathlib import Path
import json
from pydantic import BaseModel,Field
from tools.short_sentence_match import match_medical_text
from tools.text_utils import  is_measurement_paragraph
from tools.medical_preprocessor import  preprocess_text
from tools.Get_Attributes import get_all_illness_descriptions,extract_orientation
from tools.entity_merge import merge_part
BASE_DIR = Path(__file__).resolve().parent
config_path = BASE_DIR / 'config' / 'config.ini'
conf = configparser.ConfigParser()
conf.read(config_path,encoding='utf-8')
Bilateral_ORGANS=conf.get("orientation", "bilateral_organs")
SINGLE_ORGANS=conf.get("orientation", "single_organs")
tipwords = conf.get("sentence", "tipwords")
deny_words = conf.get("positive", "deny_words")
stop_pattern = conf.get("sentence", "stop_pattern")
Ignore_sentence = conf.get("clean", "Ignore_sentence")

STRONG_DEPENDENCY_KEYWORDS=conf.get("sentence", "STRONG_DEPENDENCY_KEYWORDS").split('|')
BOILERPLATE_KEYWORDS=conf.get("sentence", "BOILERPLATE_KEYWORDS")
#读取知识图谱
report_kg_df = pd.read_excel(BASE_DIR / 'config' / 'knowledgegraph.xlsx')
title_kg_df = pd.read_excel(BASE_DIR / 'config' / 'knowledgegraph_title.xlsx')
nomalMeasure = pd.read_excel(BASE_DIR / 'config' / 'Normal_measurement.xlsx')
#读取忽略词典
ignore_reports_path = BASE_DIR / 'config' / 'ignore_reports.json'
ignore_conditions = []
if ignore_reports_path.exists():
    try:
        with open(ignore_reports_path, 'r', encoding='utf-8') as f:
            ignore_conditions = json.load(f)
    except Exception as e:
        print(f"读取忽略报告配置文件失败: {e}")

# 预编译所有正则表达式
ignore_keywords = conf.get("clean", "ignore_keywords")
ignore_keywords_pattern = re.compile(rf"{ignore_keywords}", flags=re.I)

stopwords = conf.get("clean", "stopwords")
stopwords_pattern = re.compile(rf"{stopwords}", flags=re.I)

absolute_norm = conf.get("positive", "absolute_norm")
absolute_norm_set = set(absolute_norm.split("|"))

absolute_illness = conf.get("positive", "absolute_illness").split("|")
absolute_illness_set = set(absolute_illness)

NormKeyWords = conf.get("positive", "NormKeyWords")
NormKey_pattern = re.compile(NormKeyWords, flags=re.I)

illness_words = conf.get("positive", "illness_words")
illness_pattern = re.compile(illness_words, flags=re.I)

sole_words = conf.get("positive", "sole_words")
sole_words_set = set(sole_words.split("|"))

punctuation=conf.get("clean", "punctuation")
punctuation_pattern = re.compile(punctuation)
# 测量值提取的正则表达式（预编译）
measure_pattern = re.compile(
    r'(\d+(?:\.\d+)?(?:mm|cm|m|\*|×))(?![a-zA-Z])|(\d+(?:\.\d+)?(?:毫米|厘米|米))',
    flags=re.I
)
percent_pattern = re.compile(r'[\d\.]+(?:%|％)')
volume_pattern = re.compile(r'(\d+(?:\.\d+)?(?:ml|毫升))(?![a-zA-Z])', flags=re.I)

# 预处理正常测量值表
nomalMeasure = pd.read_excel(BASE_DIR / 'config' / 'Normal_measurement.xlsx')

# 将正常测量值转换为更高效的数据结构
normal_measure_rules = []
for index, row in nomalMeasure.iterrows():
    if isinstance(row['关键词'], str):
        normal_measure_rules.append({
            'pattern': re.compile(rf"{row['关键词']}", flags=re.I),
            'property': row['属性'],
            'min_val': row['最小值'],
            'max_val': row['最大值']
        })



class Report(BaseModel):
    """报告医生的数据结构.<br>
    ConclusionStr：报告结论<br>
    ReportStr:报告描述<br>
    modality:设备类型<br>
    StudyPart:检查条目名称<br>
    Sex:性别<br>
    applyTable:申请单，由既往史、临床症状、主诉、现病史拼合<br>
    """

    ConclusionStr: str= Field(title="报告结论",examples= ["""
    1.左乳术后改变，双肺及胸膜多发结节，部分较前饱满、稍大；肝内多发低密度影较前增多、范围增大；右侧心膈角稍大淋巴结。结合病史均考虑转移瘤可能性大。
    2.右肺少许炎症，右侧胸膜增厚，较前进展。右侧少量胸腔积液较前稍多。双肺少许慢性炎症同前。主动脉及冠脉硬化，心包积液大致同前。
    3.胆囊结石，胆囊炎可能，胆囊底部局限性粘膜增厚。
    4.附见：胸12及腰5椎体呈楔形改变；左侧锁骨胸骨端低密度结节。胸骨密度不均。双肾多发囊肿；双肾小结石或钙化灶；双侧肾上腺结节状增粗。
            """])   
    ReportStr: str= Field(title="报告描述",examples=["""
    “左乳腺癌术后多发转移”复查， 与2023/12/8日片比较(号码:0002093420):
    左乳缺如。右肺见少许斑片状模糊影与胸膜牵拉，右侧胸膜增厚，较前略进展；两肺多发结节，部分较前饱满、增大，结节较小增强扫描无法评估。右肺中叶及左肺上叶下舌段见斑片影及条索影，双下肺少许条索影。气管、支气管管腔完整，管壁光滑无增厚，管腔未见狭窄或阻塞。双肺门、纵隔见数枚淋巴结，同前；心影稍大，心包积液同前；主动脉、冠脉多发钙化。右侧腋窝多小淋巴结大致同前。右侧心膈角稍大淋巴结。右侧胸腔少量积液较前增多。
        附见：胸骨见少许斑点状高密度影，所见大致同前；左侧锁骨胸骨端见结节状低密度影，周边可见硬化边，大致同前。胸12、腰5椎体呈楔形改变，大致同前。
        肝内见多发稍低密度影，边界不清，肝左叶为著，，较前病灶增多、范围增大，增强扫描呈动脉期明显不均匀强化，其内多发无强化坏死区。肝内、外胆管未见明显扩张。胆囊形态、大小未见明确异常，腔内见类圆形高密度影，径约25×14mm；胆囊底部见一枚结节样突起约10mm，大致同前。胆囊壁增厚强化；胆囊窝脂肪间隙清晰。胰腺形态、大小未见明确异常，内见钙化灶，胰管未见明显扩张；胰腺周围脂肪间隙清晰；脾形态、大小、密度未见明确异常；胆、胰、脾增强扫描未见明显异常强化。腹膜后见多发稍大淋巴结。未见腹水征。
        附见：双侧肾上腺结节状增粗，以左侧内肢为著，增强扫描呈明显均匀强化。两肾见多发囊状稍低密度影，增强扫描未见明显强化。双肾见点样致密影。
            """])   
    modality:str= Field(title="设备类型",examples=['CT','MR','DX','DR','MG']) #
    StudyPart: str= Field(title="检查条目名称",examples=['CT胸部/肺平扫+增强,CT上腹部/肝胆/脾/胰平扫+增强'])  #
    Sex: str= Field(title="性别",examples=['男','女'])  #
    applyTable: str= Field(default="",title="申请单",examples=["""
                    病史:主诉:左膝外伤后肿痛不适近3个月，要求复查。
                    诊断:膝关节损伤"""])  #



def build_unified_processor(kg_dataframe: pd.DataFrame):
    """
    从完整的Excel知识图谱DataFrame构建一个统一的、信息丰富的Flashtext抽取器。
    【修正版】: 此版本正确区分了"层级关系"和"词义歧义"，确保父节点被正确识别，
    且只有末端实体（leaf node）才会被视为潜在的歧义词。

    Args:
        kg_dataframe: 从Excel文件读取的完整知识图谱DataFrame。

    Returns:
        一个配置完成的、统一的 KeywordProcessor 实例。
    """
    # 1. 使用 defaultdict(set) 来预聚合关键词的所有可能性。
    # set能自动处理因多个子节点共享同一个父节点而产生的重复信息。
    keyword_aggregator = defaultdict(set)
    
    level_columns = ['一级部位', '二级部位', '三级部位', '四级部位', '五级部位', '六级部位']

    # 2. 【新增】预计算父节点的坐标范围，以复现原GetLevelDic函数的逻辑
    # 这可以确保父节点（如"颅骨"）的坐标范围是其所有子节点坐标的并集。
    axis_cache = {}
    for level_idx, col in enumerate(level_columns):
        # 按当前列的实体名称分组
        grouped = kg_dataframe.groupby(col)
        for name, group in grouped:
            if pd.notna(name):
                # 确保name是字符串类型后再进行split操作
                name_str = str(name) if not isinstance(name, str) else name
                # 计算该实体所有出现情况下的最小起始坐标和最大终止坐标
                min_start = group['起始坐标'].min()
                max_end = group['终止坐标'].max()
                # 缓存结果，键为(层级, 实体主名称)
                axis_cache[(level_idx, name_str.split('|')[0])] = (min_start, max_end)

    # 3. 遍历知识图谱的每一行，为每个层级构建关键词和链接信息
    for index, row in kg_dataframe.iterrows():
        partlist = []
        for col in level_columns:
            if pd.notna(row[col]):
                partlist.append(row[col].split('|')[0])
            else:
                break
        
        if not partlist:
            continue

        # 4. 遍历当前行的每一个层级，为父节点和子节点都创建关键词映射
        for i in range(len(partlist)):
            # a. 当前层级的子路径，例如 ('颅脑', '颅骨')
            current_sub_path = tuple(partlist[:i+1])
            
            # b. 当前层级的坐标。如果是末端实体，直接用行数据；如果是父节点，用缓存数据。
            if i == len(partlist) - 1: # 是末端实体
                axis = (row['起始坐标'], row['终止坐标'])
            else: # 是父节点
                axis = axis_cache.get((i, partlist[i]), (None, None))

            # c. 当前层级的分类，跟随当前行的定义
            # category = row['分类']

            # d. 将当前层级的标准化信息打包成元组
            link_info = (current_sub_path, axis)

            # e. 获取当前层级的所有同义词，它们都将作为关键词
            current_level_synonyms = row[level_columns[i]].split('|')
            for keyword in current_level_synonyms:
                if keyword:
                    keyword_aggregator[keyword].add(link_info)

    # 5. 创建单一的 KeywordProcessor 实例并填充数据
    unified_processor = KeywordProcessor()
    for keyword, all_link_infos_set in keyword_aggregator.items():
        unified_processor.add_keyword(keyword, list(set(all_link_infos_set)))
        
    # print(f"Unified processor built. Total keywords: {len(unified_processor)}.")
    return unified_processor

report_processor = build_unified_processor(report_kg_df)
title_processor = build_unified_processor(title_kg_df)
# 定义处理每块Dataframe记录的函数


@lru_cache(maxsize=2560)
def find_measure(text: str) -> Tuple[float, float, float, bool]:
    """
    提取测量值（优化版）
    
    Args:
        text: 输入文本
    
    Returns:
        (最大长度值mm, 百分比, 体积ml, 是否阳性)
    """
    if not text:
        return 0.0, 0.0, 0.0, False
    
    # 标准化文本
    text = text.replace("*", "×")
    
    # 初始化返回值
    max_val = 0.0
    percent = 0.0
    volume = 0.0
    positive = False
    
    # 1. 提取长度测量值
    mes = measure_pattern.findall(text)
    if mes:
        positive = True
        for match in mes:
            # match是一个元组，提取非空的值
            value_str = match[0] if match[0] else match[1]
            if not value_str:
                continue
                
            # 提取数值和单位
            value = 0.0
            if 'cm' in value_str.lower() or '厘米' in value_str:
                try:
                    num_str = re.search(r'[\d.]+', value_str).group()
                    value = float(num_str) * 10  # 转换为mm
                except:
                    pass
            elif 'mm' in value_str.lower() or '毫米' in value_str:
                try:
                    num_str = re.search(r'[\d.]+', value_str).group()
                    value = float(num_str)
                except:
                    pass
            elif 'm' in value_str.lower() or '米' in value_str:
                try:
                    num_str = re.search(r'[\d.]+', value_str).group()
                    value = float(num_str) * 1000  # 转换为mm
                except:
                    pass
                    
            max_val = max(max_val, value)
    
    # 2. 提取百分比值
    percent_match = percent_pattern.search(text)
    if percent_match:
        positive = True
        try:
            percent = float(percent_match.group()[:-1]) / 100
        except:
            pass
    
    # 3. 提取体积值
    volume_match = volume_pattern.search(text)
    if volume_match:
        positive = True
        try:
            volume_str = volume_match.group()
            num_str = re.search(r'[\d.]+', volume_str).group()
            volume = float(num_str)
        except:
            pass
    
    # 4. 根据正常值范围判断阳性
    if positive:  # 只有在找到测量值时才判断
        for rule in normal_measure_rules:
            if rule['pattern'].search(text):
                if rule['property'] == "长度" and max_val > 0:
                    if rule['min_val'] <= max_val <= rule['max_val']:
                        positive = False
                        break
                elif rule['property'] == "百分比" and percent > 0:
                    if rule['min_val'] <= percent <= rule['max_val']:
                        positive = False
                        break
                elif rule['property'] == "体积" and volume > 0:
                    if rule['min_val'] <= volume <= rule['max_val']:
                        positive = False
                        break
    
    return max_val, percent, volume, positive


def get_positive(item: Dict) -> Tuple[bool, float, float, float]:
    """
    判断句子阳性阴性（优化版）
    
    Args:
        item: 包含short_sentence和illness的字典
    
    Returns:
        (是否阳性, 测量值, 百分比, 体积)
    """
    # 1. 检查是否包含忽略关键词
    short_sentence = item.get('short_sentence', '')
    if ignore_keywords_pattern.search(short_sentence):
        return False, 0.0, 0.0, 0.0
    
    # 2. 提取测量值并判断
    measure, percent, volume, positive = find_measure(short_sentence)
    if measure > 0 or percent > 0 or volume > 0:
        return positive, measure, percent, volume
    
    # 3. 基于疾病描述判断
    illness_text = item.get('illness', '')
    if not illness_text:
        return False, 0.0, 0.0, 0.0
    
    # 预处理疾病文本
    illness_text = re.sub(r',现|,拟|,考虑', '', illness_text)
    illness_list = [x.strip() for x in re.split('[,，]', illness_text) if x.strip()]
    
    for illness in illness_list:
        if not illness:
            continue
            
        # 去除停用词
        illness_cleaned = stopwords_pattern.sub('', illness).strip()
        illness_cleaned = punctuation_pattern.sub('', illness_cleaned)
        if not illness_cleaned:
            continue
        
        # 4. 检查是否为绝对阳性词
        if illness_cleaned in absolute_illness_set:
            return True, 0.0, 0.0, 0.0
        
        # 5. 检查是否为绝对正常词
        if illness_cleaned in absolute_norm_set:
            continue
        
        # 6. 检查是否匹配疾病模式
        if illness_pattern.search(illness_cleaned):
            return True, 0.0, 0.0, 0.0
        
        # 7. 检查是否为单独的疾病词
        if illness_cleaned in sole_words_set:
            return True, 0.0, 0.0, 0.0
        
        # 8. 如果不匹配正常关键词且长度大于1，判断为阳性
        if len(illness_cleaned) > 1 and not NormKey_pattern.search(illness_cleaned):
            return True, 0.0, 0.0, 0.0
    
    return False, 0.0, 0.0, 0.0


# ============ 批量处理优化（可选） ============

def process_positive_batch(items: List[Dict]) -> List[Dict]:
    """
    批量处理所有实体的阳性判断
    
    Args:
        items: 实体列表
    
    Returns:
        更新后的实体列表
    """
    for item in items:
        positive, measure, percent, volume = get_positive(item)
        item['positive'] = positive
        item['measure'] = measure
        item['percent'] = percent
        item['volume'] = volume
    
    return items

def match_short_sentence_batch(items: List[Dict]) -> List[Dict]:
    """
    批量处理所有实体的原始短句匹配
    
    Args:
        items: 实体列表
    
    Returns:
        更新后的实体列表
    """
    for item in items:
        result = match_medical_text(item)
        item['original_short_sentence'] =result['clause'] 

    
    return items

@lru_cache(maxsize=2560)
def Extract_Keywords(text:str,version:str='报告',modality:str='MR'):
    """Extract keywords from text using the unified processor."""
    if not text or pd.isna(text):
        return []
    if version=='报告':
        results = report_processor.extract_keywords(str(text), span_info=True)
    elif version=='标题':
        results= title_processor.extract_keywords(str(text), span_info=True)
    else:
        raise ValueError("Invalid version specified. Use '报告' or '标题'.")
    
    entity_anchors = []
    for k in results:
        # k的格式为 (keyword_info, start, end)
        keyword_info = k[0]
        start = k[1]
        end = k[2]
        partlist=[x[0] for x in keyword_info]
        axis =[x[1] for x in keyword_info]
        position=partlist[0][-1]
        keyword = text[start:end]  # 直接从文本中提取匹配的部分
        entity_anchors.append({
            'keyword': keyword,
            'axis': axis,
            'start': start,
            'end': end,
            "partlist": partlist,
            "position": position
        })
    return entity_anchors


def resplit_sentence_by_entities(
    long_sentence: str,
    original_sentence:str,
    entity_anchors: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    实体锚定与再分句函数（V8 - 简化重构版）。

    该函数遵循“分割-过滤-合并”的原则，将一个长句智能地切分为
    一个或多个干净、聚焦的“虚拟短句”列表。

    Args:
        long_sentence (str): 一个由粗糙标点分句产生的长句。
        entity_anchors (List[Dict[str, Any]]): 实体锚点列表。

    Returns:
        List[Dict[str, Any]]: 一个字典列表，每个字典是增加了上下文键的锚点。
    """
    # --- 1. 定义所需关键词 ---
    # STRONG_DEPENDENCY_KEYWORDS = [
    #     "压迫", "侵犯", "推移", "导致", "引起", 
    #     "累及", "位于", "伴", "合并", "并发"
    # ]
    # BOILERPLATE_KEYWORDS = [
    #     "请结合临床", "建议复查", "建议进一步检查", "随访复查", 
    #     "临时报告", "详请贵科阅片", "密切结合临床", "必要时"
    # ]
    
    # --- 2. [重构] 核心逻辑：分割-过滤-合并 ---
    def _get_positioned_virtual_sentences():
        # 步骤2a：分割 (Split)
        # 使用更简洁的分割方式，直接按标点切分，不保留分隔符。
        clauses = re.split(r'[，,；;]', long_sentence)
        
        # 步骤2b：过滤 (Filter)
        # 移除“无实体”且“是口水话”的子句。
        # 为了进行实体检查，我们需要知道每个子句在原始长句中的位置。
        filtered_clauses = []
        current_pos = 0
        for clause in clauses:
            clause_text = clause.strip()
            if not clause_text: continue

            clause_start = long_sentence.find(clause_text, current_pos)
            if clause_start == -1: continue
            clause_end = clause_start + len(clause_text)
            current_pos = clause_end

            has_entity = any(
                max(clause_start, anchor['start']) < min(clause_end, anchor['end'])
                for anchor in entity_anchors
            )
            is_boilerplate = re.search(BOILERPLATE_KEYWORDS,clause_text,flags=re.IGNORECASE)

            if has_entity or not is_boilerplate:
                filtered_clauses.append((clause_text, clause_start))
        
        if not filtered_clauses:
            return []

        # 步骤2c：合并 (Merge)
        # 将存在依存关系的相邻子句重新合并。
        merged_sentences = [(filtered_clauses[0][0], filtered_clauses[0][1])] # (text, start_pos)
        for i in range(1, len(filtered_clauses)):
            current_clause_text, _ = filtered_clauses[i]
            
            # 判断：如果当前子句以强依存关键词开头，则它应该依附于前一个句子。
            if any(current_clause_text.startswith(keyword) for keyword in STRONG_DEPENDENCY_KEYWORDS):
                prev_text, prev_start = merged_sentences[-1]
                merged_text = prev_text + ',' + current_clause_text
                merged_sentences[-1] = (merged_text, prev_start)
            else:
                merged_sentences.append(filtered_clauses[i])
        
        # 步骤2d：最终清洗
        final_positioned_sentences = []
        for text, start in merged_sentences:
            cleaned_text = re.sub(r'[，；。,\s]+$', '', text).strip()
            if cleaned_text:
                final_positioned_sentences.append((cleaned_text, start))

        return final_positioned_sentences

    # --- 3. 执行核心逻辑 ---
    positioned_virtual_sentences = _get_positioned_virtual_sentences()

    # --- 4. 映射虚拟短句回每个锚点 ---
    annotated_anchors = []
    for anchor in entity_anchors:
        new_anchor = anchor.copy()
        found_sentence = ""
        for vs_text, vs_start in reversed(positioned_virtual_sentences):
            if anchor['start'] >= vs_start:
                found_sentence = vs_text
                break
        mat=re.search(new_anchor['keyword'], found_sentence)
        if mat:
            new_anchor['start'] = mat.start()
            new_anchor['end'] = mat.end()

        new_anchor['sentence_start'] = long_sentence.find(found_sentence)
        new_anchor['long_sentence'] = long_sentence
        new_anchor['short_sentence'] = found_sentence
        new_anchor['original_sentence'] = original_sentence
        annotated_anchors.append(new_anchor)

    return annotated_anchors

def text_extrac_process(report_text,version='报告', modality='CT',add_info=None):
    """
    处理报告文本，提取关键词并进行实体锚定和句子拆分。
    """

    processed_text=preprocess_text(report_text, version, modality=None) # 预处理文本，返回预处理后的句子列表


    result=[]
    for index,text in enumerate(processed_text):
        original,preprocessed=text.values()
        anchors=Extract_Keywords(preprocessed,version,modality) # 提取关键词
        anchors=resplit_sentence_by_entities(preprocessed,original, anchors) #基于实体拆分子句
        anchors=[{**anchor, "sentence_index": index} for anchor in anchors] # 添加句子索引
        result.extend(anchors)
    result=sorted(result, key=lambda x: (x['sentence_index'],x['sentence_start'], x['start']))

    result=extract_orientation(result) # 提取方向信息
    result=get_all_illness_descriptions(result) # 获取所有疾病描述
    result=process_positive_batch(result)  #获取测量值和判断阳性
    result = disambiguate_entities(result, add_info)    #歧义消解
    result = merge_part(result, True if version=="标题" else False) # 合并部位信息
    result=match_short_sentence_batch(result) #匹配原始短句
    return result

def filter_paragraphs(report_text: str) -> str :
    """
    过滤段落，去重测量型描述点的段落。
    
    Args:
        paragraphs: 段落文本
    
    Returns:
        过滤后的段落。
    """
    paragraphs = re.split("\n\r" ,report_text)
    filtered_paragraphs = [paragraph for paragraph in paragraphs if paragraph.strip() and not is_measurement_paragraph(paragraph)]
    return "\n".join(filtered_paragraphs)
def report_extrac_process(ReportTxt: Report):
    """
    输入： 
       输入报告对象ReportTxt，包含结论、描述、检查部位等字段.
    功能：
    将报告的描述与结论字段的自然语言转换为结构化字段

    输出：
    partlist：六级部位列表；position：标准化部位名称；keyword：原始部位名称；
    illness：疾病描述；positive：阳性/阴性 ；measure：轴向测量值；percent：百分比值；
    short_sentence：独立意义短句,long_sentence：预处理后的长句；original_sentence：原始长句；
    sentence_index：长句在报告中的索引；sentence_start：短句在长句中的起始位置；start：关键词在短句中的起始位置；end：关键词在短句中的结束位置
    """
    studypart_analyze = text_extrac_process(ReportTxt.StudyPart, version="标题",modality=ReportTxt.modality) if ReportTxt.StudyPart else []
    result=[]
    Conclusion_analyze = text_extrac_process(filter_paragraphs(ReportTxt.ConclusionStr), add_info=[
        s['axis'] for s in studypart_analyze]) if ReportTxt.ConclusionStr else []
    Report_analyze = text_extrac_process(filter_paragraphs(ReportTxt.ReportStr), add_info=[
        s['axis'] for s in studypart_analyze]) if ReportTxt.ReportStr else []
    if len(Report_analyze)>0:
        Report_analyze=[{**dic, 'Field': 'description'} for dic in Report_analyze]
    if len(Conclusion_analyze)>0:
        Conclusion_analyze=[{**dic, 'Field': 'Conclusion'} for dic in Conclusion_analyze]
    result.extend(Report_analyze)
    result.extend(Conclusion_analyze)
    return result

# --- 使用示例 ---
def simple_example():
    # 示例1：包含强依存关系（应合并）和并列关系（应切分）的复杂长句
    long_sentence_1 = "肝门区可见肿块，压迫胆总管上段，胆囊未见增大，脂肪肝，请结合临床。双肺未见异常密度，支气管通畅。心脏增大，心腔密度减低，主动脉钙化。建议复查。胸廓入口水平见食道软组织结节。"
    long_sentence_1 = """
左侧基底节及侧脑室边缘旁见斑片状长T1长T2信号影
    """

    startTime = time.time()
    result = text_extrac_process(long_sentence_1)
    
    print("--- 示例 1 ---")
    print(f"输入长句: {long_sentence_1},耗时{time.time()-startTime:.2f}秒")
    print(f"输出的虚拟短句列表:{len(result)}")

    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    simple_example()
    # result_df=process_corpus_excel_files()
    # # 将结果保存到Excel文件
    # output_file = BASE_DIR / 'processed_copus' / 'processed_report_data'
    # (result_df[:int(len(result_df)/2)]).to_excel(str(output_file) + "1.xlsx", index=False)
    # (result_df[int(len(result_df)/2):]).to_excel(str(output_file) + "2.xlsx", index=False)


