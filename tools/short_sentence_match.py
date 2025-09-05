import re
from difflib import SequenceMatcher

def split_into_clauses(sentence):
    """
    将句子切分为多个子句
    
    Args:
        sentence: 待切分的句子
    
    Returns:
        list: 子句列表，每个元素是 (子句内容, 开始位置)
    """
    punctuation_pattern = r'[，,。；;！？、\n\r]'
    
    # 找到所有标点的位置
    splits = list(re.finditer(punctuation_pattern, sentence))
    
    # 构建子句列表，包含位置信息
    clauses = []
    start = 0
    for split in splits:
        clause = sentence[start:split.start()].strip()
        if clause:  # 跳过空子句
            clauses.append((clause, start))
        start = split.end()
    
    # 添加最后一个子句（如果有）
    if start < len(sentence):
        clause = sentence[start:].strip()
        if clause:
            clauses.append((clause, start))
    
    # 如果没有标点，整句作为一个子句
    if not clauses:
        clauses = [(sentence.strip(), 0)]
    
    return clauses


def find_best_matching_clause_for_single(short_clause, original_sentence):
    """
    为单个短句在原句中找到最佳匹配的子句（原有逻辑）
    
    Args:
        short_clause: 单个短子句
        original_sentence: 原始句子
    
    Returns:
        tuple: (最匹配的子句, 相似度分数, 位置索引)
    """
    
    # 第0层：完全匹配检查
    if short_clause in original_sentence:
        start_idx = original_sentence.index(short_clause)
        return (short_clause, 1.0, start_idx)
    
    # 第1层：切割子句
    original_clauses = split_into_clauses(original_sentence)
    
    # 第2层：快速筛选 + 相似度计算
    short_chars = set(short_clause)
    best_match = None
    best_score = -1
    
    for clause, position in original_clauses:
        # 字符重合度快速筛选
        clause_chars = set(clause)
        overlap_ratio = len(short_chars & clause_chars) / len(short_chars) if short_chars else 0
        
        # 重合度太低，仍然计算但可以快速跳过
        if overlap_ratio < 0.3:  # 极低的重合度
            # 给一个很低的分数，避免完全计算
            similarity = overlap_ratio * 0.5
        else:
            # 第3层：使用difflib计算相似度
            matcher = SequenceMatcher(None, short_clause, clause)
            
            # 先用quick_ratio快速估算
            quick_score = matcher.quick_ratio()
            if quick_score < 0.3:  # 极低分数，使用快速估算值
                similarity = quick_score
            else:
                # 精确计算相似度
                similarity = matcher.ratio()
                
                # 第4层：位置权重优化（医学报告开头词汇更重要）
                # 比较前5个字符的相似度
                prefix_len = min(5, len(short_clause), len(clause))
                if prefix_len > 0:
                    prefix_matcher = SequenceMatcher(None, 
                                                    short_clause[:prefix_len], 
                                                    clause[:prefix_len])
                    prefix_similarity = prefix_matcher.ratio()
                    if prefix_similarity > 0.8:
                        similarity = min(1.0, similarity + 0.05)  # 轻微加权
        
        # 更新最佳匹配
        if similarity > best_score:
            best_score = similarity
            best_match = (clause, similarity, position)
    
    return best_match


def find_best_matching_clause(short_sentence, original_sentence):
    """
    快速找到与short_sentence相似度最高的子句或多个子句组合
    支持多子句匹配
    
    Args:
        short_sentence: 短句（可能包含多个子句）
        original_sentence: 原始句子
    
    Returns:
        tuple: (最匹配的子句, 相似度分数, 位置索引)
    """
    
    # 第0层：完全匹配检查
    if short_sentence in original_sentence:
        start_idx = original_sentence.index(short_sentence)
        return (short_sentence, 1.0, start_idx)
    
    # 第1层：检查short_sentence是否包含多个子句
    short_clauses = split_into_clauses(short_sentence)
    
    # 如果short_sentence只有一个子句，使用原有的单子句匹配逻辑
    if len(short_clauses) == 1:
        return find_best_matching_clause_for_single(short_sentence, original_sentence)
    
    # 第2层：多子句匹配逻辑
    # 为每个短子句在原句中找到最佳匹配
    matched_clauses = []
    total_similarity = 0
    used_positions = set()  # 记录已使用的原句子句位置，避免重复匹配
    
    for short_clause, _ in short_clauses:
        best_match = None
        best_score = -1
        
        # 在原句中寻找当前短子句的最佳匹配
        clause, similarity, position = find_best_matching_clause_for_single(short_clause, original_sentence)
        
        # 检查该位置是否已被使用（避免多个短子句匹配到同一个原子句）
        original_clauses = split_into_clauses(original_sentence)
        original_clause_at_pos = None
        for orig_clause, orig_pos in original_clauses:
            if orig_pos == position:
                original_clause_at_pos = orig_clause
                break
        
        if original_clause_at_pos and position not in used_positions:
            matched_clauses.append(clause)
            total_similarity += similarity
            used_positions.add(position)
        elif original_clause_at_pos:  # 位置已被使用，但仍然可以包含该匹配
            matched_clauses.append(clause)
            total_similarity += similarity * 0.7  # 降低权重
    
    # 第3层：组合匹配结果
    if matched_clauses:
        # 将匹配的子句按在原句中的位置顺序排列
        original_clauses = split_into_clauses(original_sentence)
        sorted_matches = []
        
        for orig_clause, orig_pos in original_clauses:
            if orig_clause in matched_clauses:
                sorted_matches.append(orig_clause)
        
        # 组合成完整的匹配句子
        combined_match = "，".join(sorted_matches)
        avg_similarity = total_similarity / len(short_clauses) if len(short_clauses) > 0 else 0
        
        # 找到组合匹配句子在原句中的起始位置
        start_position = original_sentence.find(sorted_matches[0]) if sorted_matches else 0
        
        return (combined_match, avg_similarity, start_position)
    
    # 如果没有找到匹配，返回最相似的单个子句
    return find_best_matching_clause_for_single(short_sentence, original_sentence)


def match_medical_text(data_dict):
    """
    主函数：处理医学文本匹配，支持多子句匹配
    
    Args:
        data_dict: 包含short_sentence和original_sentence的字典
    
    Returns:
        dict: 匹配结果，包含是否为多子句匹配的信息
    """
    short = data_dict["short_sentence"]
    original = data_dict["original_sentence"]
    
    clause, similarity, position = find_best_matching_clause(short, original)
    
    # 检查是否进行了多子句匹配
    short_clauses = split_into_clauses(short)
    is_multi_clause = len(short_clauses) > 1
    
    return {
        "clause": clause,
        "similarity": similarity,
        "position": position,
        "is_multi_clause": is_multi_clause,
        "short_clauses_count": len(short_clauses),
        "short_clauses": [clause[0] for clause in short_clauses]  # 只返回子句内容，不包含位置
    }


# 测试代码
if __name__ == "__main__":
    # 测试数据
    test_data = {
        "short_sentence": "双侧胸腔少量积液与前大致相仿，",
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