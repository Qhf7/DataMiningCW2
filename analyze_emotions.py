import json
from collections import Counter

try:
    # 打开并读取文件
    records = []
    with open('Dataset/final_dataset.json', 'r') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    print(f"解析错误: {line[:50]}...")
    
    print(f"成功解析 {len(records)} 条记录")
    
    # 提取所有情绪
    all_emotions = []
    for record in records:
        if 'feelings' in record:
            all_emotions.extend(record['feelings'])
    
    # 统计情绪
    emotion_counts = Counter(all_emotions)
    
    # 按出现次数降序排序
    sorted_emotions = emotion_counts.most_common()
    
    # 打印情绪统计结果
    print("\n情绪分布统计表:")
    print("=" * 40)
    print(f"{'情绪类型':<25} {'出现次数':<10} {'百分比':<10}")
    print("-" * 40)
    
    total_emotions = sum(emotion_counts.values())
    for emotion, count in sorted_emotions:
        percentage = (count / total_emotions) * 100
        print(f"{emotion:<25} {count:<10} {percentage:.1f}%")
    
    print("=" * 40)
    
    # 计算总情绪数和前三种情绪的百分比
    print(f"\n总记录数: {len(records)}")
    print(f"总情绪标记数: {total_emotions}")
    print(f"平均每条记录的情绪标记数: {total_emotions / len(records):.2f}")
    
    print("\n前三种最常见的情绪及其百分比:")
    for emotion, count in sorted_emotions[:3]:
        percentage = (count / total_emotions) * 100
        print(f"{emotion}: {count} ({percentage:.1f}%)")

except Exception as e:
    print(f"发生错误: {str(e)}") 