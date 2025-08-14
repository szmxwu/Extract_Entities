#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from functools import lru_cache
from typing import Optional, List, Tuple


class MedicalExpander:
    """医学术语扩展类，用于展开医学报告中的缩写形式。"""
    
    # 句子分割模式
    SENTENCE_PATTERN = r"[?？。；;\n\r]"
    
    # 脊柱相关关键词
    SPINE_KEYWORDS = (r"椎|横突|棘|脊|黄韧带|肋|[^a-zA-Z]L[^a-zA-Z]|颈|骶|尾骨|"
                     r"骨(?!.*[信号|FLAIR])|腰大肌|腰[1-5]|胸[1-9]|隐裂|腰化|"
                     r"骶化|胸化|项韧带|纵韧带|腰骶"
                     r"|([^c|C|l|L|t|T|s|S])(\d{1,2})")
    
    def __init__(self):
        """初始化医学扩展器，预编译所有正则表达式模式。"""
        self._compile_patterns()
        self._create_need_expand_pattern()
    
    def _compile_patterns(self):
        """编译所有正则表达式模式。"""
        # 脊柱扩展相关模式
        self.patterns = {
            'spine1': re.compile(r'([^颈胸腰骶尾])(\d{1,2})[、|,|，|及|和](\d{1,2})([颈|胸|腰|骶|尾])(?!.*段)', re.I),
            'spine2': re.compile(r'([^颈胸腰骶尾/])(\d{1,2})[、|,|，|及|和]([颈|胸|腰|骶|尾])(\d{1,2})(?!.*段)', re.I),
            'spine3': re.compile(r'([^颈胸腰骶尾])(\d{1,2})(/\d{1,2})[、|,|，|及|和](\d{1,2})(/\d{1,2})([颈|胸|腰|骶|尾])', re.I),
            'spine4': re.compile(r'([^颈胸腰骶尾])(\d{1,2})(/\d{1,2})[、|,|，|及|和]([颈|胸|腰|骶|尾])(\d{1,2})(/\d{1,2})', re.I),
            'spine5': re.compile(r'([腰|胸|颈|骶|尾|c|l|t|s])(\d{1,2})(/\d{1,2})?[、|,|，|及|和](\d{1,2})', re.I),
            'spine6': re.compile(r'([腰|胸|颈|骶|尾|c|l|t|s]\d{1,2})(/\d{1,2})?[,|，]([腰|胸|颈|骶|尾|c|l|t|s]\d{1,2})', re.I),
            'disk_range': re.compile(r"([胸|腰|颈|骶|c|t|l|s])(\d{1,2})/([胸|腰|颈|骶|c|t|l|s])?(\d{1,2})-([胸|腰|颈|骶|c|t|l|s])?(\d{1,2})/([胸|腰|颈|骶|c|t|l|s])?(\d{1,2})(?!.*段)", re.I),
            'spine_range': re.compile(r"([胸|腰|颈|骶|c|t|l|s])(\d{1,2})-([胸|腰|颈|骶|c|t|l|s])?(\d{1,2})(椎体)?(?!.*段)", re.I),
            'c_abbr': re.compile(r'(^|[^a-zA-Z])C([1-8])(?!.*段)', re.I),
            't_abbr': re.compile(r'(^|[^a-zA-Z长短低高等脂水])T(\d{1,2})(?!.*[段_信号压黑为呈示a-zA-Z])', re.I),
            't_spine': re.compile(r'(^|[^a-zA-Z])T(\d{1,2})椎', re.I),
            't_number': re.compile(r'(^|[^a-zA-Z])T([3-9]|10|11|12)(?!.*[MN])', re.I),
            's_abbr': re.compile(r'(^|[^a-zA-Z])S([1-5])(?![段a-zA-Z0-9])', re.I),
            'dot_fix': re.compile(r"([a-zA-Z\u4e00-\u9fa5])\.([a-zA-Z\u4e00-\u9fa5])", re.I),
        }
        
        # 肋骨扩展相关模式
        self.rib_pattern = re.compile(
            r"(双侧|双|[左右]|)"   # Group 1: 可选前缀
            r"(第)"                # Group 2: '第'
            r"([\d、，]+)"         # Group 3: 数字列表或范围起始
            r"(?:-(\d+))?"        # Group 4: 可选的范围结束
            r"([前后腋]?)"         # Group 5: 可选中缀
            r"(肋骨?)"             # Group 6: '肋骨' 或 '肋'
            r"([^，。、\s]+)"      # Group 7: 后缀
        )
        
        # 句子分割模式
        self.sentence_split_pattern = re.compile(self.SENTENCE_PATTERN)
        
        # 脊柱关键词模式
        self.spine_keywords_pattern = re.compile(self.SPINE_KEYWORDS)
    
    def _create_need_expand_pattern(self):
        """创建用于检测是否需要扩展的模式。"""
        # 检测是否包含需要扩展的医学术语
        patterns_to_check = [
            r"[颈胸腰骶尾cCtTlLsS]\d{1,2}",  # 脊柱编号
            r"\d{1,2}/\d{1,2}",              # 椎间盘表示
            r"\d{1,2}-\d{1,2}",              # 范围表示
            r"第\d+[、，-]",                  # 肋骨编号
            r"肋骨?",                        # 肋骨关键词
            self.SPINE_KEYWORDS               # 脊柱关键词
        ]
        
        combined_pattern = "|".join(patterns_to_check)
        self.need_expand_pattern = re.compile(combined_pattern, re.I)
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _get_last_spine_number(spine_type: str) -> int:
        """获取特定脊柱类型的最大编号。"""
        spine_map = {
            '颈': 7, 'c': 7, 'C': 7,
            '胸': 12, 't': 12, 'T': 12,
            '腰': 5, 'l': 5, 'L': 5,
            '骶': 5, 's': 5, 'S': 5,
            '尾': 3
        }
        return spine_map.get(spine_type, 3)
    
    @lru_cache(maxsize=512)
    def _preprocess_text(self, text: str) -> str:
        """预处理文本，清洗和标准化。"""
        # 清洗空白字符
        text = re.sub(r"(）|\)) ", r"\1\n", text)
        text = re.sub(r"[ \xa0\x7f\u3000]", "", text)
        text = re.sub(r"^[\n\t\r]+\d[\.|、]", "", text)
        text = re.sub(r"\b(\d+[.、])(?!\d)([\u4e00-\u9fffa-zA-Z][\u4e00-\u9fffa-zA-Z，。；]*)\b", 
                     r"\2", text, flags=re.X | re.IGNORECASE)
        
        # 修复句号问题
        text = self.patterns['dot_fix'].sub(r"\1。\2", text)
        
        return text
    
    def _split_sentences(self, text: str) -> List[str]:
        """将文本分割成句子。"""
        sentence_ends = [match.start() for match in self.sentence_split_pattern.finditer(text)]
        
        if not sentence_ends:
            sentence_ends = [len(text)]
        if sentence_ends[0] != 0:
            sentence_ends = [0] + sentence_ends
        if sentence_ends[-1] != len(text):
            sentence_ends.append(len(text))
        
        sentences = []
        for i in range(len(sentence_ends) - 1):
            sentence = text[sentence_ends[i]:sentence_ends[i + 1]]
            if sentence:
                sentences.append(sentence)
        
        return sentences
    
    @lru_cache(maxsize=512)
    def _normalize_spine_abbreviations(self, text: str) -> str:
        """标准化脊柱缩写形式。"""
        if self.spine_keywords_pattern.search(text):
            text = self.patterns['c_abbr'].sub(r'\1颈\2', text)
            text = self.patterns['t_abbr'].sub(r'\1胸\2', text)
            text = self.patterns['t_spine'].sub(r'\1胸\2', text)
            text = self.patterns['s_abbr'].sub(r'\1骶\2', text)
        
        text = self.patterns['t_number'].sub(r'\1胸\2', text)
        return text
    
    @lru_cache(maxsize=256)
    def _expand_spine_ranges(self, text: str) -> str:
        """展开脊柱范围表示（如 L1-L5）。"""
        matches = self.patterns['spine_range'].findall(text)
        if not matches:
            return text
        
        for group in matches:
            try:
                new_parts = []
                spine_type1, start_num, spine_type2, end_num, suffix = group
                
                if spine_type1 == spine_type2 or not spine_type2:
                    last = int(end_num)
                else:
                    last = self._get_last_spine_number(spine_type1)
                
                start = int(start_num)
                end = int(end_num)
                
                # 生成第一段脊柱
                for i in range(start, last + 1):
                    new_parts.append(f"{spine_type1}{i}椎体")
                
                # 生成第二段脊柱（如果存在）
                if spine_type1 != spine_type2 and spine_type2:
                    for i in range(1, end + 1):
                        new_parts.append(f"{spine_type2}{i}椎体")
                
                old_str = f"{spine_type1}{start_num}-{spine_type2}{end_num}{suffix}"
                new_str = "、".join(new_parts)
                text = text.replace(old_str, new_str)
            except (ValueError, IndexError):
                continue
        
        return text
    
    @lru_cache(maxsize=256)
    def _expand_disk_ranges(self, text: str) -> str:
        """展开椎间盘范围表示（如 L1/2-L5/S1）。"""
        matches = self.patterns['disk_range'].findall(text)
        if not matches:
            return text
        
        for group in matches:
            try:
                new_parts = []
                s1, n1, s2, n2, s3, n3, s4, n4 = group
                
                if s1 == s3 or not s3:
                    last = int(n3)
                else:
                    last = self._get_last_spine_number(s1)
                
                start = int(n1)
                end = int(n3)
                
                # 生成第一段椎间盘
                for i in range(start, last + 1):
                    new_parts.append(f"{s1}{i}/{i+1}")
                
                # 生成第二段椎间盘（如果存在）
                if s1 != s3 and s3:
                    for i in range(1, end + 1):
                        new_parts.append(f"{s3}{i}/{i+1}")
                
                # 替换特殊情况
                new_str = "、".join(new_parts)
                new_str = re.sub(r"[颈|c]7/8", "颈7/胸1", new_str, flags=re.I)
                new_str = re.sub(r"[胸|t]12/13", "胸12/腰1", new_str, flags=re.I)
                new_str = re.sub(r"[腰|l]5/6", "腰5/骶1", new_str, flags=re.I)
                
                old_str = f"{s1}{n1}/{s2}{n2}-{s3}{n3}/{s4}{n4}"
                text = text.replace(old_str, new_str)
            except (ValueError, IndexError):
                continue
        
        return text
    
    @lru_cache(maxsize=256)
    def _expand_spine_dots(self, text: str) -> str:
        """展开脊柱顿号形式（如 L1、2、3）。"""
        text = self.patterns['spine1'].sub(r"\1\4\2、\4\3", text)
        text = self.patterns['spine3'].sub(r"\1\6\2\3、\6\4\5", text)
        
        for _ in range(10):  # 最多迭代10次
            new_text = self.patterns['spine2'].sub(r"\1\3\2、\3\4", text)
            new_text = self.patterns['spine4'].sub(r"\1\4\2\3、\4\5\6", new_text)
            new_text = self.patterns['spine5'].sub(r"\1\2\3、\1\4", new_text)
            new_text = self.patterns['spine6'].sub(r"\1\2、\3", new_text)
            
            if new_text == text:
                break
            text = new_text
        
        return text
    
    @lru_cache(maxsize=256)
    def _expand_rib_abbreviations(self, text: str) -> str:
        """展开肋骨缩写形式。"""
        if "肋" not in text:
            return text
        
        def replace_match(match):
            prefix = match.group(1) or ""
            marker = match.group(2)  # "第"
            num_part = match.group(3)
            range_end = match.group(4)
            infix = match.group(5) or ""
            rib_word = match.group(6) if match.group(6) else "肋骨"
            suffix = match.group(7) or ""
            
            numbers_to_expand = []
            
            if range_end:
                try:
                    start_num = int(num_part)
                    end_num = int(range_end)
                    if start_num <= end_num:
                        numbers_to_expand = list(range(start_num, end_num + 1))
                    else:
                        return match.group(0)
                except ValueError:
                    return match.group(0)
            else:
                try:
                    numbers_to_expand = [
                        int(n) for n in re.split(r'[、，]', num_part) 
                        if n.strip().isdigit()
                    ]
                except ValueError:
                    return match.group(0)
            
            if not numbers_to_expand:
                return match.group(0)
            
            expanded_parts = [
                f"{prefix}{marker}{num}{infix}{rib_word}{suffix}"
                for num in numbers_to_expand
            ]
            
            return "，".join(expanded_parts)
        
        return self.rib_pattern.sub(replace_match, text)
    
    def expand_all(self, text: str) -> str:
        """执行所有医学术语扩展。
        
        Args:
            text: 需要扩展的医学文本
            
        Returns:
            扩展后的文本
        """
        # 预处理
        text = self._preprocess_text(text)
        
        # 分句处理
        sentences = self._split_sentences(text)
        processed_sentences = []
        
        for sentence in sentences:
            # 标准化缩写
            sentence = self._normalize_spine_abbreviations(sentence)
            
            # 展开脊柱范围
            sentence = self._expand_spine_ranges(sentence)
            
            # 展开椎间盘范围
            sentence = self._expand_disk_ranges(sentence)
            
            # 展开顿号形式
            sentence = self._expand_spine_dots(sentence)
            
            # 展开肋骨缩写
            sentence = self._expand_rib_abbreviations(sentence)
            
            processed_sentences.append(sentence)
        
        return "\n".join(processed_sentences)
    
    def process_text(self, text: str) -> str:
        """处理文本的主入口方法（兼容旧接口）。
        
        Args:
            text: 需要处理的医学文本
            
        Returns:
            处理后的文本
        """
        if self.need_expand_pattern.search(text):
            return self.expand_all(text)
        return text


# 使用示例
if __name__ == "__main__":
    # 创建扩展器实例
    expander = MedicalExpander()
    
    # 测试用例
    test_cases = [
        "L1/2-L5/S1椎间盘突出",
        "右第5、6前肋骨折，左第7、8后肋不全性骨折",
        "C1-C7椎体未见明显异常",
        "T3、4、5椎体压缩性骨折",
        "腰3-5椎体退行性改变"
    ]
    
    for test_str in test_cases:
        print(f"原文: {test_str}")
        
        # 方法1：使用 need_expand_pattern 检查
        if expander.need_expand_pattern.search(test_str):
            result = expander.expand_all(test_str)
            print(f"扩展后: {result}")
        else:
            print("无需扩展")
        
        # 方法2：直接使用 process_text（内部会检查）
        # result = expander.process_text(test_str)
        # print(f"处理后: {result}")
        
        print("-" * 50)