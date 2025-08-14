import re
from typing import List, Dict, Any, Tuple
import configparser
from pathlib import Path
from collections import defaultdict
import json
import pandas as pd
from itertools import chain
from functools import lru_cache
BASE_DIR = Path(__file__).resolve().parent.parent
config_path = BASE_DIR / 'config' / 'config.ini'
# nomalMeasure = pd.read_excel(BASE_DIR / 'config' / 'Normal_measurement.xlsx')
conf = configparser.ConfigParser()
conf.read(config_path,encoding='utf-8')
Bilateral_ORGANS=conf.get("orientation", "bilateral_organs")
SINGLE_ORGANS=conf.get("orientation", "single_organs")
tipwords = conf.get("sentence", "tipwords")
deny_words = conf.get("positive", "deny_words")
stop_words = conf.get("sentence", "stop_pattern")
absolute_norm = conf.get("positive", "absolute_norm")
absolute_illness = conf.get("positive", "absolute_illness").split("|")
NormKeyWords = conf.get("positive", "NormKeyWords")
NormKey_pattern=re.compile(NormKeyWords,flags=re.I) 
illness_words = conf.get("positive", "illness_words")
illness_pattern=re.compile(illness_words,flags=re.I)
sole_words = conf.get("positive", "sole_words")
sole_words_set = set(sole_words.split("|"))
Ignore_sentence = conf.get("clean", "Ignore_sentence")
stopwords = conf.get("clean", "stopwords")
ignore_keywords = conf.get("clean", "ignore_keywords")
# 连接词正则（可自行扩充）
CONN_RE = re.compile(r'(伴有|伴|合并|并)')
RULE_KEYWORDS = {
    "deny_words": deny_words,
    "tipwords": tipwords,
    "stopwords": stop_words
}

def get_illness_description(
    target_entity: Dict[str, Any],
    all_entities_in_sentence: List[Dict[str, Any]],
    sentence_text: str,
    rule_keywords: Dict[str, str]
) -> Tuple[str, str]:
    """
    仅依赖实体在列表中的顺序，不依赖 start/end 绝对值。
    """
    deny_pat = re.compile(rule_keywords.get("deny_words", ""))
    tip_pat  = re.compile(rule_keywords.get("tipwords", ""))
    stop_pat = re.compile(rule_keywords.get("stopwords", ""))
    split_mode = ""
    # -------- 1. 找到目标在列表中的位置 --------
    try:
        target_idx = next(i for i, e in enumerate(all_entities_in_sentence)
                          if e is target_entity)
    except StopIteration:
        return "", "entity_not_found"

    # -------- 2. 预处理：方位定语/引导词句 --------
    effective_sentence = sentence_text
    effective_entities = all_entities_in_sentence.copy()

    last_tip = None
    for m in tip_pat.finditer(sentence_text):
        # 只要引导词在目标实体之前即可
        if m.end() <= all_entities_in_sentence[target_idx]['start']:
            last_tip = m
        else:
            if m.end() <=all_entities_in_sentence[min(target_idx+3,len(all_entities_in_sentence)-1)]['start']:
                return "", "entity_not_found"
            else:
                break

    if last_tip:
        # 引导词左侧、右侧都要有实体
        left_ent = any(e['end'] <= last_tip.start() for e in effective_entities)
        right_ent = any(e['start'] >= last_tip.end() for e in effective_entities)
        if left_ent and right_ent:
            cut = last_tip.end()
            effective_sentence = sentence_text[cut:].strip()
            # 重新计算实体在截断后句子中的坐标
            new_ents = []
            for e in effective_entities:
                if e['start'] >= cut:
                    new_ents.append({
                        **e,
                        'start': e['start'] - cut,
                        'end':   e['end']   - cut
                    })
            if new_ents:
                effective_entities = new_ents
                # 重新找目标索引（值相同，但列表已变）
                target_idx = next(i for i, e in enumerate(effective_entities)
                                  if e['keyword'] == target_entity['keyword'])

    # -------- 3. 定义前后文窗口 --------
    left_text  = ""
    right_text = ""
    if target_idx > 0:
        prev_end = effective_entities[target_idx - 1]['end']
        curr_start = effective_entities[target_idx]['start']
        left_text = effective_sentence[prev_end:curr_start].strip()
    else:
        left_text = effective_sentence[:effective_entities[target_idx]['start']].strip()

    if target_idx + 1 < len(effective_entities):
        curr_end = effective_entities[target_idx]['end']
        next_start = effective_entities[target_idx + 1]['start']
        right_text = effective_sentence[curr_end:next_start].strip()
    else:
        right_text = effective_sentence[effective_entities[target_idx]['end']:].strip()
    right_text = re.sub(r'[，；、。,;]', '', right_text).strip()
    edge_text=effective_sentence[effective_entities[-1]['end']:].strip()
    if len(right_text)<=2 and len(edge_text)>len(right_text) and right_text not in sole_words_set:
        right_text = edge_text
    # -------- 4. 三种模式匹配 --------
    # 4-a 否定前置
    deny_match = deny_pat.match(effective_sentence)
    if deny_match:
        suffix = re.sub(r'[。；、，,;\s]', '', right_text)
        if 0 < len(suffix) <= 2:
            illness = deny_match.group(0) + suffix
            return illness, "split_negation_pattern"

    # 4-b 倒置句
    if not right_text and left_text and not tip_pat.search(left_text):
        clean_left = re.sub(r'[，；、。,;]', '', left_text)
        if len(clean_left) > 1:
            return clean_left.strip('，；。 '), "inverted_pattern"

    # 4-c 默认后置
    illness = re.sub(r'^[，；、。,;]', '', right_text).strip()
    illness = re.sub(r'[。；、，,;]$', '', illness)
    illness = stop_pat.sub('', illness).strip()
    return illness, "default_postfix_pattern"



def _reindex_entities(ents: List[Dict[str, Any]], sentence: str) -> List[Dict[str, Any]]:
    """
    利用 keyword 在 sentence 中重新定位，返回新的 start/end（0-based）
    """
    new_ents = []
    for e in ents:
        keyword = e['keyword']
        # 允许 orientation 前缀，如“双肺”→“肺”
        pattern = re.escape(keyword)
        if e.get('orientation'):
            pattern = re.escape(e['orientation']) + r'?' + pattern
        m = re.search(pattern, sentence)
        if not m:
            # 兜底：原样返回
            new_ents.append({**e, 'start': e['start'], 'end': e['end']})
            continue
        new_ents.append({
            **e,
            'start': m.start(),
            'end': m.end()
        })
    return new_ents


# ---------------- 主函数 ----------------
def get_all_illness_descriptions(
    entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    输入：List[Dict]，含 keyword/orientation/start/end/short_sentence/sentence_index
    输出：同结构，新增 key "illness"
    """
    # 1. 按 short_sentence 分组
    by_sentence = defaultdict(list)
    for ent in entities:
        by_sentence[ent['short_sentence']].append(ent)

    # 2. 结果表 (sentence_index, keyword) 唯一
    result_map: Dict[Tuple[int, str], str] = {}

    for sentence, ents_in_sent in by_sentence.items():
        # 2-a 重新编号实体坐标
        ents_reidx = _reindex_entities(ents_in_sent, sentence)
        # 2-b 切分连接词
        chunks = _split_by_connector(ents_reidx, sentence)

        # 2-c 逐子句抽取
        for sub_sent, sub_ents in chunks:
            for ent in sub_ents:
                illness, _ = get_illness_description(
                    ent,
                    sub_ents,
                    sub_sent,
                    RULE_KEYWORDS
                )
                key = (ent['sentence_index'], ent['keyword'])
                result_map[key] = illness

    # 3. 回填
    return [
        {**ent, 'illness': result_map.get((ent['sentence_index'], ent['keyword']), '')}
        for ent in entities
    ]


# ------------- 辅助：按连接词切分 -------------
def _split_by_connector(
    entities: List[Dict[str, Any]],
    sentence: str
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    conn_matches = list(CONN_RE.finditer(sentence))
    if not conn_matches:
        return [(sentence, entities)]

    cuts = [0]
    for m in conn_matches:
        left_ok = any(e['end'] <= m.start() for e in entities)
        right_ok = any(e['start'] >= m.end() for e in entities)
        if left_ok and right_ok:
            cuts.append(m.start())
            cuts.append(m.end())
    cuts.append(len(sentence))
    cuts = sorted(set(cuts))

    chunks = []
    for s0, s1 in zip(cuts, cuts[1:]):
        sub_sent = sentence[s0:s1].strip()
        if not sub_sent:
            continue
        sub_raw = [e for e in entities if s0 <= e['start'] < s1]
        # 子句内再重编号
        sub_reidx = _reindex_entities(sub_raw, sub_sent)
        # 平移坐标到 0-based
        offset = s0
        for e in sub_reidx:
            e['start'] -= offset
            e['end'] -= offset
        chunks.append((sub_sent, sub_reidx))
    return chunks if len(chunks) > 1 else [(sentence, entities)]







def extract_orientation(data: List[Dict]) -> List[Dict]:
    """
    提取医学句子中的方位词（左/右/双/两）
    :param data: 输入的解剖实体列表
    :return: 添加 orientation 字段后的列表
    """
    # 深拷贝避免修改原数据
    result = [item.copy() for item in data]
    result = sorted(result, key=lambda x: (x['sentence_index'], x['start']))
    # 第一轮：显式方位词提取
    for item in result:
        keyword = item["keyword"]
        short_sentence = item["short_sentence"]

        # 单器官直接跳过
        if re.search(SINGLE_ORGANS,keyword):
            item["orientation"] = ""
            continue

        # 初始化
        orientation = ""

        # 从 keyword 开始向前查找方位词
        keyword_start_in_sentence = short_sentence.find(keyword)
        if keyword_start_in_sentence != -1:
            prefix = short_sentence[:keyword_start_in_sentence]
            match = re.search(r'(左|右|双|两)(?![侧位])', prefix[::-1])
            if match:
                orientation = match.group(1)[::-1]

        # 若前方未找到，向后查找括号内方位词
        if not orientation:
            keyword_end_in_sentence = keyword_start_in_sentence + len(keyword)
            suffix = short_sentence[keyword_end_in_sentence:]
            match = re.search(r'[(（](左|右|双|两)[侧部]*?[)）]', suffix)
            if match:
                orientation = match.group(1)

        item["orientation"] = orientation.replace("两", "双")

    # 第二轮：上下文推理补充
    for i, item in enumerate(result):
        if re.search(rf"{Bilateral_ORGANS}"," ".join(chain.from_iterable(item['partlist']))) is None or item["orientation"]:
            continue

        # 向前查找最近非空 orientation
        for j in range(i - 1, -1, -1):
            if result[j]["orientation"]:
                item["orientation"] = result[j]["orientation"]
                break

        # 若仍未找到，向后查找最多2个非空 orientation
        if not item["orientation"]:
            for j in range(i + 1, min(i + 3, len(result))):
                if result[j]["orientation"]:
                    item["orientation"] = result[j]["orientation"]
                    break

    return result

# --- 使用示例 ---
if __name__ == '__main__':
    # 示例1: 倒置句
    # sentence1 = "血肿位于肾脏。"
    # entities1 = [{'keyword': '肾脏', 'start': 4, 'end': 6,'short_sentence':"血肿位于肾脏"}]
    # illness1=get_all_illness_descriptions(entities1)
    # print(f"句子: '{entities1[0]['short_sentence']}'")
    # print(f"  -> 模式: {illness1}\n")

    # # 示例2: 否定前置句
    # sentence2 = "未见颅脑异常。"
    # entities2 = [{'keyword': '颅脑', 'start': 2, 'end': 4,'short_sentence':"未见颅脑异常"}]
    # illness2=get_all_illness_descriptions(entities2)
    # print(f"句子: '{entities2[0]['short_sentence']}'")
    # print(f"  -> 模式: {illness2}\n")

    # 示例3: 正常的方位定语句 (Bug修复验证)
    sentence3 = "胸廓入口水平见食道软组织结节。"
    entities3 = [{'keyword': '胸廓入口', 'sentence_index': 0,'start': 0, 'end': 6,'short_sentence':"胸廓入口水平见食道软组织结节。"}, {'keyword': '食道', 'sentence_index': 0,'start': 7, 'end': 9,'short_sentence':"胸廓入口水平见食道软组织结节。"}]
    # 测试目标为“食道”
    illness3=get_all_illness_descriptions(entities3)
    print(f"句子: '{entities3[0]['short_sentence']}'")
    print(f"  -> 模式: {illness3}\n")

    # # 示例4: 标准后置句 (Bug修复验证)
    # sentence4 = "肝脏增大"
    # entities4 = [{'keyword': '肝脏', 'start': 0, 'end': 2, 'short_sentence':"肝脏增大"}]
    # illness4=get_all_illness_descriptions(entities4)
    # print(f"句子: '{entities4[0]['short_sentence']}'")
    # print(f"  -> 模式: {illness4}\n")

    # # 示例5: 引导词后无实体（应归为标准后置句）
    # sentence5 = "肝脏显示：未见异常密度。"
    # entities5 = [{'keyword': '肝脏', 'start': 0, 'end': 2, 'short_sentence':"肝脏显示：未见异常密度"}]
    # illness5=get_all_illness_descriptions(entities5)
    # print(f"句子: '{entities5[0]['short_sentence']}'")
    # print(f"  -> 模式: {illness5}\n")
    
    # # 示例6: 复杂的实体间描述 (Bug修复验证)
    # sentence6 = "肝脏可见一枚结节压迫胆总管上段。"
    # entities6 = [{'keyword': '肝脏', 'start': 0, 'end': 2,'short_sentence':"肝脏可见一枚结节压迫胆总管上段。"}, {'keyword': '胆总管', 'start': 10, 'end': 13,'short_sentence':"肝脏可见一枚结节压迫胆总管上段。"}]
    # illness6=get_all_illness_descriptions(entities6)
    # print(f"句子: '{entities6[0]['short_sentence']}'")
    # print(f"  -> 模式: {illness6}\n")

    #     # 示例6: 复杂的实体间描述 
    # demo = [
    #     {'keyword': '支气管', 'start': 0, 'end': 3, 'short_sentence': "支气管扩张伴左肺感染"},
    #     {'keyword': '肺',     'start': 7, 'end': 8, 'short_sentence': "支气管扩张伴左肺感染"}
    # ]
    # illness6=get_all_illness_descriptions(demo)
    # print(f"句子: '{demo[0]['short_sentence']}'")
    # print(f"  -> 模式: {illness6}\n")
    
    # demo = [
    #     {'keyword': '支气管', 'start': 0, 'end': 3, 'short_sentence': "支气管扩张伴感染"},
    # ]
    # illness6=get_all_illness_descriptions(demo)
    # print(f"句子: '{demo[0]['short_sentence']}'")
    # print(f"  -> 模式: {illness6}\n")

    # demo=[{'keyword': '肝门', 'orientation': '', 'start': 0, 'end': 2, 'sentence_index': 0, 'short_sentence': '肝门区可见肿块，压迫胆总管上段'}, 
    #       {'keyword': '胆总管', 'orientation': '', 'start': 10, 'end': 13, 'sentence_index': 0, 'short_sentence': '肝门区可见肿块，压迫胆总管上段'}, 
    #       {'keyword': '胆囊', 'orientation': '', 'start': 16, 'end': 18, 'sentence_index': 0, 'short_sentence': '胆囊未见增大'}, 
    #       {'keyword': '肝', 'orientation': '', 'start': 25, 'end': 26, 'sentence_index': 0, 'short_sentence': '脂肪肝'}, 
    #       {'keyword': '肺', 'orientation': '双', 'start': 1, 'end': 2, 'sentence_index': 1, 'short_sentence': '双肺未见异常密度'},
    #         {'keyword': '支气管', 'orientation': '', 'start': 9, 'end': 12, 'sentence_index': 1, 'short_sentence': '支气管通畅'}, 
    #         {'keyword': '心脏', 'orientation': '', 'start': 0, 'end': 2, 'sentence_index': 2, 'short_sentence': '心脏增大'}, 
    #         {'keyword': '心腔', 'orientation': '', 'start': 5, 'end': 7, 'sentence_index': 2, 'short_sentence': '心腔密度减低'}, 
    #         {'keyword': '主动脉', 'orientation': '', 'start': 12, 'end': 15, 'sentence_index': 2, 'short_sentence': '主动脉钙化'}]


    # extracted_data=get_all_illness_descriptions(demo)
    # print(json.dumps(extracted_data, indent=4, ensure_ascii=False))
    #     # for item in result:
    #     #     print(item)   
    # with open('output.json', 'w', encoding='utf-8') as f:
    #     json.dump(extracted_data, f, indent=4, ensure_ascii=False)