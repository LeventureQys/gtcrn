import os
import argparse

def batch_rename_files(folder_path):
    """
    批量将文件夹内文件名末尾的 _clean 替换为 _noisy（仅处理.wav文件）
    
    Args:
        folder_path: 目标文件夹路径
    """
    # 验证文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在！")
        return
    
    # 统计变量
    renamed_count = 0
    skip_count = 0
    
    # 遍历文件夹内所有文件
    for filename in os.listdir(folder_path):
        # 只处理.wav文件，且文件名包含_clean
        if filename.endswith(".wav") and "_clean" in filename:
            # 找到最后一个_clean的位置（确保只替换末尾的_clean）
            clean_pos = filename.rfind("_clean")
            if clean_pos != -1:
                # 构造新文件名：替换最后一个_clean为_noisy
                new_filename = filename[:clean_pos] + "_noisy" + filename[clean_pos + len("_clean"):]
                
                # 拼接完整路径
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)
                
                # 避免覆盖已存在的文件
                if os.path.exists(new_path):
                    print(f"跳过：新文件 '{new_filename}' 已存在，不覆盖")
                    skip_count += 1
                    continue
                
                # 执行重命名
                try:
                    os.rename(old_path, new_path)
                    print(f"已重命名：{filename} -> {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"失败：重命名 '{filename}' 时出错 - {str(e)}")
                    skip_count += 1
        else:
            skip_count += 1
    
    # 输出统计结果
    print("\n=== 重命名完成 ===")
    print(f"成功重命名：{renamed_count} 个文件")
    print(f"跳过/失败：{skip_count} 个文件")

if __name__ == "__main__":
    # 命令行参数解析（方便直接运行）
    parser = argparse.ArgumentParser(description="批量将文件夹内.wav文件名的_clean替换为_noisy")
    parser.add_argument("folder", help="目标文件夹的路径（绝对路径/相对路径均可）")
    args = parser.parse_args()
    
    # 调用重命名函数
    batch_rename_files(args.folder)