"""
medical_preprocessor.py
简化的医学报告预处理模块
支持不同版本和设备类型的规则加载
"""

import re
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from functools import lru_cache
from pathlib import Path
import json
import time
if __name__ == "__main__":
    from medical_expander import MedicalExpander
    from text_utils import *
else:
    from .medical_expander import MedicalExpander
    from .text_utils import *
# ============ 配置和常量 ============

BASE_DIR = Path(__file__).resolve().parent.parent

# 分句模式
SENTENCE_SPLIT_PATTERN = re.compile(r'[。；？;！!\n\r]+')


# 版本到文件和工作表的映射
# 版本到文件的映射
VERSION_FILE_MAP = {
    '报告': 'replace.xlsx',
    '标题': 'replace_title.xlsx',
    '申请单': 'replace_applytable.xlsx'
}

# ============ 预编译的正则表达式 ============




class MedicalPatterns:
    """预编译的医学正则表达式"""
    
    # 基础清洗
    SPACE_CLEAN = re.compile(r'[\xa0\x7f\u3000\s]+')
    PUNCT_CLEAN = re.compile(r'\s*([。；，、？！])\s*')
    Correct_STOP=re.compile(r"([a-zA-Z\u4e00-\u9fa5])\.([a-zA-Z\u4e00-\u9fa5])",flags=re.I) 
    # 医学标识符转换
    SPINE_C = re.compile(r'(^|[^a-zA-Z])C([1-8])(?!.*段)',flags=re.I) 
    SPINE_T1 = re.compile(r'(^|[^a-zA-Z长短低高等脂水])T(\d{1,2})(?!.*[段_信号压黑为呈示a-zA-Z])',flags=re.I) 
    SPINE_T2 = re.compile(r'(^|[^a-zA-Z])T(\d{1,2})椎',flags=re.I) 
    SPINE_T3=re.compile(r'(^|[^a-zA-Z])T([3-9]|10|11|12)(?!.*[MN])',flags=re.I) 
    SPINE_S = re.compile(r'(^|[^a-zA-Z])S([1-5])(?![段a-zA-Z0-9])',flags=re.I) 
    SPINE_WORDS=re.compile(r'椎|横突|棘|脊|黄韧带|肋|[^a-zA-Z]L[^a-zA-Z]|颈|骶|尾骨|骨(?!.*[信号|FLAIR])|腰大肌|腰[1-5]|胸[1-9]|隐裂|腰化|骶化|胸化|项韧带|纵韧带|腰骶')
    # 椎体范围扩展
    SPINE_RANGE = re.compile(r'([颈胸腰骶尾])(\d{1,2})[、,，及和](\d{1,2})(?!.*段)')
    SPINE_MULTI = re.compile(r'([颈胸腰骶尾])(\d{1,2})-(\d{1,2})(?!.*段)')
    
    # 椎间盘扩展
    DISK_SIMPLE = re.compile(r'([颈胸腰骶CTLS])(\d{1,2})/(\d{1,2})')
    DISK_RANGE = re.compile(r'([颈胸腰骶CTLS])(\d{1,2})/(\d{1,2})-([颈胸腰骶CTLS])?(\d{1,2})/(\d{1,2})')
    
    # 肋骨扩展
    RIB_RANGE = re.compile(
        r'(双侧|双|[左右]?)'
        r'(第)?'
        r'(\d{1,2})'
        r'(?:[-、，,及和至到])'
        r'(\d{1,2})'
        r'([前后腋]?)'
        r'(肋骨?)'
    )
    RIB_MULTI = re.compile(
        r'(双侧|双|[左右]?)'
        r'(第)?'
        r'([\d、，,]+)'
        r'([前后腋]?)'
        r'(肋骨?)'
    )


# ============ 替换规则管理 ============

class ReplacementRules:
    """替换规则管理器"""
    
    def __init__(self, 
                 excel_dir: Optional[Path] = None,
                 version: str = '报告',
                 modality: Optional[str] = None):
        """
        初始化替换规则
        
        Args:
            excel_dir: Excel文件所在目录路径
            version: 版本类型（报告/标题/申请单）
            modality: 设备类型（CT/MR/DR等），如果指定则从对应sheet读取
        """
        self.simple_rules = {}  # 简单文本替换
        self.regex_rules = []   # 正则替换规则
        self.priority_rules = {} # 高优先级规则
        self.version = version
        self.modality = modality
        
        if excel_dir and excel_dir.exists():
            self._load_from_excel(excel_dir)
        else:
            self._load_default_rules()
    
    def _load_default_rules(self):
        """加载默认规则"""
        # 高频医学缩写
        self.priority_rules = {
            '增强扫描': '增强',
        }
        
        # 常用同义词
        self.simple_rules = {
            '未见明显': '未见',
            '未见确切': '未见',
            '可见': '见',
            '大致': '',
            '约': '',
            '左侧': '左',
            '右侧': '右',
            '双侧': '双',
            '两侧': '双',
        }
        
        # 正则替换
        self.regex_rules = [
            (re.compile(r'^\d+[、.]'), ''),  # 删除句首编号
            (re.compile(r'【.*?】'), ''),     # 删除标记
            (re.compile(r'\(.*?\)'), ''),     # 删除括号内容
            (re.compile(r'（.*?）'), ''),     # 删除中文括号内容
        ]
    
    def _load_from_excel(self, excel_dir: Path):
        """从Excel加载规则"""
        try:
            # 获取对应版本的文件
            file_name = VERSION_FILE_MAP.get(self.version, VERSION_FILE_MAP['报告'])
            excel_path = excel_dir / file_name
            
            if not excel_path.exists():
                print(f"规则文件不存在: {excel_path}")
                self._load_default_rules()
                return
            
            # 如果指定了modality，尝试从modality sheet读取
            if self.modality is not None:
                self._load_with_modality(excel_path, self.modality)
            else:
                # 否则加载默认的直接替换和正则替换规则
                self._load_default_sheets(excel_path)
                
        except Exception as e:
            print(f"加载Excel规则失败: {e}")
            self._load_default_rules()
    
    def _load_with_modality(self, excel_path: Path, modality: str):
        """加载特定设备类型的规则"""
        try:
            # 尝试读取modality对应的sheet
            if modality=="DR":
                modality = "DX"
            df = pd.read_excel(excel_path, sheet_name=modality)
            self._parse_rules_dataframe(df)
            print(f"成功从 '{excel_path.name}' 加载设备类型 '{modality}' 的规则")
        except Exception as e:
            # 如果modality sheet不存在或读取失败，回退到sheet 0
            print(f"无法加载设备类型 '{modality}' 的规则: {e}")
            print(f"回退到默认规则（sheet 0）")
            try:
                df = pd.read_excel(excel_path, sheet_name=0)
                self._parse_rules_dataframe(df)
            except Exception as e2:
                print(f"加载默认规则也失败: {e2}")
                self._load_default_rules()

    def _load_default_sheets(self, excel_path: Path):
        """加载默认的直接替换和正则替换规则"""
        # 加载直接替换规则（sheet 0）
        try:
            df_direct = pd.read_excel(excel_path, sheet_name=0)
            self._parse_direct_rules(df_direct)
        except Exception as e:
            print(f"无法加载直接替换规则: {e}")
        
        # 加载正则替换规则（sheet 1）
        try:
            df_regex = pd.read_excel(excel_path, sheet_name=1)
            self._parse_regex_rules(df_regex)
        except Exception as e:
            print(f"无法加载正则替换规则: {e}")

    def _load_modality_rules(self, excel_dir: Path, modality: str):
        """加载特定设备类型的规则"""
        try:
            
            modality_file = excel_dir / f'replace_{modality.lower()}.xlsx'
            
            if modality_file.exists():
                # 如果存在设备特定文件，使用它
                df_direct = pd.read_excel(modality_file, sheet_name=0)
                self._parse_direct_rules(df_direct)
                
                try:
                    df_regex = pd.read_excel(modality_file, sheet_name=1)
                    self._parse_regex_rules(df_regex)
                except:
                    pass  # 正则规则是可选的
                    
                print(f"成功加载设备类型 '{modality}' 的规则文件")
            else:
                # 如果modality文件不存在，回退到version规则
                print(f"未找到设备类型 '{modality}' 的规则文件，使用版本规则")
                self._load_version_rules(excel_dir, self.version)
                
        except Exception as e:
            print(f"无法加载设备类型 '{modality}' 的规则: {e}")
            self._load_version_rules(excel_dir, self.version)
    
    def _load_version_rules(self, excel_dir: Path, version: str):
        """加载特定版本的规则"""
        file_info = VERSION_FILE_MAP.get(version, VERSION_FILE_MAP['报告'])
        excel_path = excel_dir / file_info['file']
        
        if not excel_path.exists():
            print(f"规则文件不存在: {excel_path}")
            self._load_default_rules()
            return
        
        # 加载直接替换规则
        try:
            df_direct = pd.read_excel(excel_path, sheet_name=file_info['direct_sheet'])
            self._parse_direct_rules(df_direct)
        except Exception as e:
            print(f"无法加载版本 '{version}' 的直接替换规则: {e}")
        
        # 加载正则替换规则
        try:
            df_regex = pd.read_excel(excel_path, sheet_name=file_info['regex_sheet'])
            self._parse_regex_rules(df_regex)
        except Exception as e:
            print(f"无法加载版本 '{version}' 的正则替换规则: {e}")
    
    def _parse_rules_dataframe(self, df: pd.DataFrame):
        """解析通用规则DataFrame（用于modality）"""
        for _, row in df.iterrows():
            if pd.isna(row.get('原始值')):
                continue
                
            original = str(row['原始值'])
            replacement = str(row.get('替换值', '')) if not pd.isna(row.get('替换值')) else ''
            is_regex = row.get('IsRegex', False)
            
            if is_regex:
                try:
                    pattern = re.compile(original, re.IGNORECASE)
                    self.regex_rules.append((pattern, replacement))
                except Exception as e:
                    print(f"正则表达式编译失败 '{original}': {e}")
            else:
                # 根据使用频率分类
                if len(original) <= 4:  # 短词通常是高频词
                    self.priority_rules[original] = replacement
                else:
                    self.simple_rules[original] = replacement
    
    def _parse_direct_rules(self, df: pd.DataFrame):
        """解析直接替换规则"""
        for _, row in df.iterrows():
            if pd.isna(row.get('原始值')):
                continue
                
            original = str(row['原始值'])
            replacement = str(row.get('替换值', '')) if not pd.isna(row.get('替换值')) else ''
            
            # 根据长度分类
            if len(original) <= 4:
                self.priority_rules[original] = replacement
            else:
                self.simple_rules[original] = replacement
    
    def _parse_regex_rules(self, df: pd.DataFrame):
        """解析正则替换规则"""
        for _, row in df.iterrows():
            if pd.isna(row.get('原始值')):
                continue
                
            original = str(row['原始值'])
            replacement = str(row.get('替换值', '')) if not pd.isna(row.get('替换值')) else ''
            
            try:
                pattern = re.compile(original, re.IGNORECASE)
                self.regex_rules.append((pattern, replacement))
            except Exception as e:
                print(f"正则表达式编译失败 '{original}': {e}")





# ============ 主预处理器 ============

class MedicalPreprocessor:
    """医学报告预处理器"""
    
    def __init__(self, 
                 rules_dir: Optional[Path] = None,  # 改为目录路径
                 version: str = '报告',
                 modality: Optional[str] = None,
                 enable_cache: bool = True):
        """
        初始化预处理器
        
        Args:
            rules_dir: 替换规则Excel文件所在目录路径
            version: 版本类型（报告/标题/申请单）
            modality: 设备类型（CT/MR/DR等），如果指定则优先使用
            enable_cache: 是否启用缓存
        """
        self.patterns = MedicalPatterns()
        self.version = version
        self.modality = modality
        self.rules = ReplacementRules(rules_dir, version, modality)  # 传入目录路径
        self.expander = MedicalExpander()
        self.enable_cache = enable_cache
        
        # 预编译检测模式
        self.need_expand_pattern = self.expander.need_expand_pattern
        
        # 如果启用缓存，清空缓存（因为规则可能不同）
        if self.enable_cache:
            self._process_sentence_cached.cache_clear()
    
    def split_sentences(self, text: str) -> List[Tuple[str, int]]:
        """
        分句函数
        
        Args:
            text: 输入文本
            
        Returns:
            [(句子, 起始位置), ...]
        """
        if not text:
            return []
        
        # 使用分句模式分割
        sentences = []
        last_end = 0
        
        for match in SENTENCE_SPLIT_PATTERN.finditer(text):
            sentence = text[last_end:match.start()].strip()
            if sentence:
                sentences.append((sentence, last_end))
            last_end = match.end()
        
        # 处理最后一个句子
        if last_end < len(text):
            sentence = text[last_end:].strip()
            if sentence:
                sentences.append((sentence, last_end))
        
        return sentences if sentences else [(text, 0)]
    
    def process_sentence(self, sentence: str) -> str:
        """
        处理单个句子
        
        Args:
            sentence: 输入句子
            
        Returns:
            处理后的句子
        """
        if not sentence or len(sentence) < 2:
            return sentence
        
        # 如果启用缓存，使用缓存版本
        if self.enable_cache:
            # 缓存键包含version和modality
            cache_key = (sentence, self.version, self.modality)
            return self._process_sentence_cached(cache_key)
        return self._process_sentence_impl(sentence)
    
    @lru_cache(maxsize=1024)
    def _process_sentence_cached(self, cache_key: Tuple[str, str, Optional[str]]) -> str:
        """缓存版本的句子处理"""
        sentence, _, _ = cache_key
        return self._process_sentence_impl(sentence)
    
    def _process_sentence_impl(self, sentence: str) -> str:
        """
        句子处理的实际实现
        """
        # 1. 基础清洗
        sentence = self.patterns.SPACE_CLEAN.sub(' ', sentence)
        sentence = self.patterns.PUNCT_CLEAN.sub(r'\1', sentence)
        sentence=clean_whitespace(sentence)
        sentence = normalize_punctuation(sentence)
        sentence = remove_numbered_prefix(sentence)
        # 2. 高优先级替换（高频词）
        for old, new in self.rules.priority_rules.items():
            sentence = sentence.replace(old, new)
        
        # 3. 医学扩展（先扩展，这样可以处理原始的L1/2-L5/S1格式）
        if self.need_expand_pattern.search(sentence):
            sentence = self.expander.expand_all(sentence)
        
        # 4. 脊柱标识符转换（扩展后再转换剩余的单独标识符）
        if self.patterns.SPINE_WORDS.search(sentence):
            sentence = self.patterns.SPINE_C.sub('\\1颈\\2',sentence)
            sentence = self.patterns.SPINE_T1.sub('\\1胸\\2',sentence)
            sentence = self.patterns.SPINE_T2.sub('\\1胸\\2',sentence)
            sentence = self.patterns.SPINE_S.sub('\\1骶\\2',sentence)
        sentence = self.patterns.SPINE_T3.sub('\\1胸\\2',sentence)
        
        # 5. 一般替换
        for old, new in self.rules.simple_rules.items():
            sentence = sentence.replace(old, new)
        
        # 6. 正则替换
        for pattern, replacement in self.rules.regex_rules:
            sentence = pattern.sub(replacement, sentence)
        
        # 7. 最终清理
        sentence = sentence.strip()
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = re.sub(r'[,，、]{2,}', '、', sentence)
        sentence=self.patterns.Correct_STOP.sub(r"\1。\2",sentence)
        return sentence
    
    def process(self, text: str) -> List[Dict[str, str]]:
        """
        主处理函数
        
        Args:
            text: 输入文本
            
        Returns:
            [{'original': str, 'preprocessed': str}, ...]
        """
        if not text:
            return []
        
        # 分句
        sentences = self.split_sentences(text)
        
        # 处理每个句子
        results = []
        for original_sentence, _ in sentences:
            if not original_sentence:
                continue
                
            # 处理句子
            processed_sentence = self.process_sentence(original_sentence)
            
            # 只有处理后不为空的句子才添加到结果
            if processed_sentence:
                results.append({
                    'original': original_sentence,
                    'preprocessed': processed_sentence
                })
        
        return results


# ============ 便捷接口 ============

# 全局预处理器缓存（根据version和modality缓存不同的实例）
_preprocessor_cache = {}

def get_preprocessor(version: str = '报告', 
                    modality: Optional[str] = None) -> MedicalPreprocessor:
    """
    获取预处理器实例（带缓存）
    
    Args:
        version: 版本类型
        modality: 设备类型
        
    Returns:
        预处理器实例
    """
    global _preprocessor_cache
    
    cache_key = (version, modality)
    
    if cache_key not in _preprocessor_cache:
        # 确定规则文件目录
        rules_dir = BASE_DIR / 'config'
        if not rules_dir.exists():
            rules_dir = BASE_DIR
        
        _preprocessor_cache[cache_key] = MedicalPreprocessor(
            rules_dir=rules_dir if rules_dir.exists() else None,
            version=version,
            modality=modality,
            enable_cache=True
        )
    
    return _preprocessor_cache[cache_key]


def preprocess_text(text: str, 
                   version: str = '报告', 
                   modality: Optional[str] = None) -> List[Dict[str, str]]:
    """
    主接口函数（支持version和modality参数）
    
    Args:
        text: 输入文本
        version: 版本类型（报告/标题/申请单）
        modality: 设备类型（CT/MR/DR等），如果指定则优先使用
        
    Returns:
        [{'original': str, 'preprocessed': str}, ...]
    """
    preprocessor = get_preprocessor(version, modality)
    return preprocessor.process(text)


def create_preprocessor(version: str = '报告', 
                       modality: Optional[str] = None,
                       **kwargs) -> MedicalPreprocessor:
    """
    创建预处理器实例
    
    Args:
        version: 版本类型（报告/标题/申请单）
        modality: 设备类型（CT/MR/DR等）
        **kwargs: 其他参数
        
    Returns:
        预处理器实例
    """
    rules_dir = kwargs.get('rules_dir', BASE_DIR / 'config')
    if not rules_dir.exists():
        rules_dir = BASE_DIR
    
    enable_cache = kwargs.get('enable_cache', True)
    
    return MedicalPreprocessor(
        rules_dir=rules_dir if rules_dir.exists() else None,
        version=version,
        modality=modality,
        enable_cache=enable_cache
    )

def test():
    # 测试文本
    test_text = """
    急诊报告： 与2023/1/7日片比较: 两肺多发斑片渗出实变影，边界不清,内见空洞形成。
    双侧胸腔少量积液与前相仿，C1-3椎体未见异常。T12、L1-3椎体压缩性骨折。
    第1-3肋骨骨折，左侧第5、6前肋不完全性骨折。L1/2-L5/S1椎间盘突出。
    颅脑CTA检查未见明显异常，MRA示血管走行自然。
    """
    
    print("=" * 60)
    print("测试不同版本的预处理")
    print("=" * 60)
    
    # 测试报告版本
    print("\n1. 报告版本：")
    print("-" * 40)
    results = preprocess_text(test_text, version='报告')
    for i, item in enumerate(results, 1):
        print(f"句子{i}:")
        print(f"  原始: {item['original'][:50]}...")
        print(f"  处理后: {item['preprocessed'][:50]}...")
    
    # 测试标题版本
    print("\n2. 标题版本：")
    print("-" * 40)
    title_text = "CT胸部平扫+增强，MRI颅脑平扫"
    results = preprocess_text(title_text, version='标题')
    for i, item in enumerate(results, 1):
        print(f"句子{i}:")
        print(f"  原始: {item['original']}")
        print(f"  处理后: {item['preprocessed']}")
    
    # 测试申请单版本
    print("\n3. 申请单版本：")
    print("-" * 40)
    apply_text = "主诉：胸痛3天。既往史：高血压病史5年。"
    results = preprocess_text(apply_text, version='申请单')
    for i, item in enumerate(results, 1):
        print(f"句子{i}:")
        print(f"  原始: {item['original']}")
        print(f"  处理后: {item['preprocessed']}")
    
    # 测试设备类型参数
    print("\n4. 测试设备类型参数（如果有CT规则）：")
    print("-" * 40)
    results = preprocess_text(test_text, version='报告')
    print(f"处理了 {len(results)} 个句子")
    
    # 性能测试
    print("\n" + "=" * 60)
    print("性能测试")
    print("=" * 60)
    
    # 创建不同版本的预处理器
    preprocessor_report = create_preprocessor(version='报告')
    preprocessor_title = create_preprocessor(version='标题')
    
    # 测试报告处理性能
    start_time = time.time()
    for _ in range(100):
        _ = preprocessor_report.process(test_text)
    elapsed = time.time() - start_time
    print(f"报告版本处理100次耗时: {elapsed:.3f}秒")
    print(f"平均每次: {elapsed * 10:.2f}ms")
    
    # 测试标题处理性能
    start_time = time.time()
    for _ in range(100):
        _ = preprocessor_title.process(title_text)
    elapsed = time.time() - start_time
    print(f"标题版本处理100次耗时: {elapsed:.3f}秒")
    print(f"平均每次: {elapsed * 10:.2f}ms")
    
    # 测试缓存效果
    print("\n缓存效果测试：")
    print("-" * 40)
    
    # 第一次处理（无缓存）
    start_time = time.time()
    _ = preprocess_text(test_text, version='报告')
    first_time = time.time() - start_time
    
    # 第二次处理（有缓存）
    start_time = time.time()
    _ = preprocess_text(test_text, version='报告')
    second_time = time.time() - start_time
    
    print(f"第一次处理: {first_time * 1000:.2f}ms")
    print(f"第二次处理（缓存）: {second_time * 1000:.2f}ms")

# ============ 测试代码 ============

if __name__ == "__main__":
    test_text="L1/2-L5/S1椎间盘突出"
    results = preprocess_text(test_text, version='报告')
    for i, item in enumerate(results, 1):
        print(f"句子{i}:")
        print(f"  原始: {item['original'][:50]}...")
        print(f"  处理后: {item['preprocessed'][:50]}...")
    test()

