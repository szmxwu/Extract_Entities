import re
from difflib import SequenceMatcher

def find_best_matching_clause(short_sentence, original_sentence):
    """
    快速找到与short_sentence相似度最高的子句
    
    Args:
        short_sentence: 短句
        original_sentence: 原始句子
    
    Returns:
        tuple: (最匹配的子句, 相似度分数, 位置索引)
    """
    
    # 第0层：完全匹配检查
    if short_sentence in original_sentence:
        start_idx = original_sentence.index(short_sentence)
        return (short_sentence, 1.0, start_idx)
    
    # 第1层：切割子句
    # 使用中文标点符号切割，保留位置信息
    punctuation_pattern = r'[，。；！？、]'
    
    # 找到所有标点的位置
    splits = list(re.finditer(punctuation_pattern, original_sentence))
    
    # 构建子句列表，包含位置信息
    clauses = []
    start = 0
    for split in splits:
        clause = original_sentence[start:split.start()]
        if clause:  # 跳过空子句
            clauses.append((clause, start))
        start = split.end()
    
    # 添加最后一个子句（如果有）
    if start < len(original_sentence):
        clause = original_sentence[start:]
        if clause:
            clauses.append((clause, start))
    
    # 如果没有标点，整句作为一个子句
    if not clauses:
        clauses = [(original_sentence, 0)]
    
    # 第2层：快速筛选 + 相似度计算
    short_chars = set(short_sentence)
    best_match = None
    best_score = -1
    
    for clause, position in clauses:
        # 字符重合度快速筛选
        clause_chars = set(clause)
        overlap_ratio = len(short_chars & clause_chars) / len(short_chars) if short_chars else 0
        
        # 重合度太低，仍然计算但可以快速跳过
        if overlap_ratio < 0.3:  # 极低的重合度
            # 给一个很低的分数，避免完全计算
            similarity = overlap_ratio * 0.5
        else:
            # 第3层：使用difflib计算相似度
            matcher = SequenceMatcher(None, short_sentence, clause)
            
            # 先用quick_ratio快速估算
            quick_score = matcher.quick_ratio()
            if quick_score < 0.3:  # 极低分数，使用快速估算值
                similarity = quick_score
            else:
                # 精确计算相似度
                similarity = matcher.ratio()
                
                # 第4层：位置权重优化（医学报告开头词汇更重要）
                # 比较前5个字符的相似度
                prefix_len = min(5, len(short_sentence), len(clause))
                if prefix_len > 0:
                    prefix_matcher = SequenceMatcher(None, 
                                                    short_sentence[:prefix_len], 
                                                    clause[:prefix_len])
                    prefix_similarity = prefix_matcher.ratio()
                    if prefix_similarity > 0.8:
                        similarity = min(1.0, similarity + 0.05)  # 轻微加权
        
        # 更新最佳匹配
        if similarity > best_score:
            best_score = similarity
            best_match = (clause, similarity, position)
    
    return best_match


def match_medical_text(data_dict):
    """
    主函数：处理医学文本匹配
    
    Args:
        data_dict: 包含short_sentence和original_sentence的字典
    
    Returns:
        dict: 匹配结果
    """
    short = data_dict["short_sentence"]
    original = data_dict["original_sentence"]
    
    clause, similarity, position = find_best_matching_clause(short, original)
    
    return {
        "clause": clause,
        "similarity": similarity,
        "position": position
    }


# 测试代码
if __name__ == "__main__":
    # 测试数据
    test_data = {
        "short_sentence": "双侧胸腔少量积液与前大致相仿",
        "original_sentence": "双侧胸腔少量积液与前相仿，双肺下叶膨胀不全，右肺下叶较前改善"
    }
    
    # 执行匹配
    result = match_medical_text(test_data)
    
    # 打印结果
    print("匹配结果：")
    print(f"最佳匹配: '{result['clause']}'")
    print(f"相似度: {result['similarity']:.3f}")
    print(f"位置: {result['position']}")
    
    # 测试更多案例
    print("\n其他测试案例:")
    test_cases = [
        {
            "short_sentence": "肺部感染",
            "original_sentence": "双肺炎症，考虑感染性病变，建议复查"
        },
        {
            "short_sentence": "心脏增大",
            "original_sentence": "心影增大，主动脉迂曲，双肺纹理增多"
        },
        {
            "short_sentence": "完全不匹配的句子",
            "original_sentence": "双侧胸腔少量积液，双肺下叶膨胀不全"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        result = match_medical_text(test)
        print(f"\n案例{i}:")
        print(f"  查找: '{test['short_sentence']}'")
        print(f"  匹配: '{result['clause']}'")
        print(f"  相似度: {result['similarity']:.3f}")
    
    # 性能测试
    import time
    
    print("\n性能测试:")
    start_time = time.perf_counter()
    for _ in range(1000):
        find_best_matching_clause(test_data["short_sentence"], 
                                 test_data["original_sentence"])
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / 1000 * 1000  # 转换为毫秒
    print(f"平均执行时间: {avg_time:.3f} 毫秒")
    print(f"单次执行时间: {avg_time*1000:.1f} 微秒")