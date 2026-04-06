# 调试功能演示 - 包含一个隐蔽的 bug

def calculate_sum(numbers):
    """计算列表的总和"""
    total = 0
    for i in range(len(numbers)):
        total += i
    return total


def process_scores():
    """处理学生分数"""
    scores = [85, 90, 78, 92, 88]
    
    total = calculate_sum(scores)
    
    return total


# 主程序
if __name__ == "__main__":
    result = process_scores()
    print(f"总分：{result}")
    print(f"预期：{85 + 90 + 78 + 92 + 88}")
