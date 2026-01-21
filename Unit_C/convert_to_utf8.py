#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将项目文件转换为 UTF-8 编码
"""

import os
import sys
from pathlib import Path

def detect_encoding(file_path):
    """检测文件编码（尝试常见编码）"""
    encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'cp1252', 'utf-16']
    
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            if not raw_data:
                return 'utf-8'
            
            # 检查 BOM
            if raw_data.startswith(b'\xef\xbb\xbf'):
                # UTF-8 with BOM
                return 'utf-8-sig'
            elif raw_data.startswith(b'\xff\xfe'):
                # UTF-16 LE
                return 'utf-16-le'
            elif raw_data.startswith(b'\xfe\xff'):
                # UTF-16 BE
                return 'utf-16-be'
            
            # 尝试各种编码
            for encoding in encodings_to_try:
                try:
                    raw_data.decode(encoding)
                    return encoding
                except (UnicodeDecodeError, LookupError):
                    continue
            
            # 如果都失败，使用 utf-8 并忽略错误
            return 'utf-8'
    except Exception as e:
        print(f"Error detecting encoding for {file_path}: {e}")
        return 'utf-8'

def convert_to_utf8(file_path, target_encoding='utf-8'):
    """将文件转换为 UTF-8 编码"""
    try:
        # 检测当前编码
        detected_encoding = detect_encoding(file_path)
        
        # 读取文件
        with open(file_path, 'r', encoding=detected_encoding, errors='replace') as f:
            content = f.read()
        
        # 如果已经是 UTF-8 且内容相同，跳过
        if detected_encoding.lower().replace('-', '').replace('_', '') == 'utf8':
            # 检查是否需要移除 BOM
            if content.startswith('\ufeff'):
                content = content[1:]  # 移除 BOM
                print(f"Removed BOM from: {file_path}")
            else:
                # 已经是 UTF-8 无 BOM，检查是否需要重写
                try:
                    with open(file_path, 'rb') as f:
                        raw = f.read()
                        # 检查是否有 BOM
                        if raw.startswith(b'\xef\xbb\xbf'):
                            # 有 BOM，需要移除
                            pass
                        else:
                            # 已经是 UTF-8 无 BOM，跳过
                            return False
                except:
                    pass
        
        # 写入 UTF-8（无 BOM）
        with open(file_path, 'w', encoding='utf-8', newline='', errors='replace') as f:
            f.write(content)
        
        if detected_encoding.lower().replace('-', '').replace('_', '') != 'utf8':
            print(f"Converted: {file_path} ({detected_encoding} -> UTF-8)")
        return True
        
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return False

def convert_directory(directory, extensions=None):
    """转换目录中的所有文件"""
    if extensions is None:
        extensions = ['.c', '.h', '.py', '.md', '.txt', '.cmake', '.bat', '.sh']
    
    converted_count = 0
    skipped_count = 0
    error_count = 0
    
    for root, dirs, files in os.walk(directory):
        # 跳过 build 目录和 .git 目录
        dirs[:] = [d for d in dirs if d not in ['build', '.git', '__pycache__', 'Debug', 'Release', 'x64']]
        
        for file in files:
            file_path = Path(root) / file
            
            # 检查文件扩展名
            if any(file_path.suffix.lower() == ext.lower() for ext in extensions):
                try:
                    if convert_to_utf8(str(file_path)):
                        converted_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(f"Failed to convert {file_path}: {e}")
                    error_count += 1
    
    print(f"\n转换完成:")
    print(f"  已转换: {converted_count} 个文件")
    print(f"  已跳过: {skipped_count} 个文件（已经是 UTF-8）")
    print(f"  错误: {error_count} 个文件")

if __name__ == '__main__':
    # 获取项目根目录
    project_root = Path(__file__).parent.absolute()
    
    print(f"开始转换项目文件为 UTF-8 编码...")
    print(f"项目目录: {project_root}")
    print()
    
    # 转换源代码目录
    directories = ['src', 'include', 'examples', 'tests', 'scripts', 'docs']
    
    for dir_name in directories:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"\n处理目录: {dir_name}")
            convert_directory(str(dir_path))
    
    # 转换根目录的文件
    print(f"\n处理根目录文件...")
    root_files = ['CMakeLists.txt', 'README.md', 'README_NEW.md']
    for file_name in root_files:
        file_path = project_root / file_name
        if file_path.exists():
            convert_to_utf8(str(file_path))
    
    # 也处理其他可能的文件
    for ext in ['.c', '.h', '.py', '.md', '.txt', '.cmake', '.bat', '.sh']:
        for file in project_root.glob(f'*{ext}'):
            if file.is_file() and file.name not in ['convert_to_utf8.py']:
                convert_to_utf8(str(file))
    
    print("\n所有文件转换完成！")

