# kwargs_detailed_example.py
# 详细讲解 **kwargs (可变关键字参数)

# ============================================================
# **kwargs 是什么？
# ============================================================
# **kwargs 允许函数接收任意数量的关键字参数
# 这些参数会被收集到一个字典 (dict) 中
# "kwargs" 是约定俗成的名字，你可以用任何名字，但 ** 是必须的

# ============================================================
# 基础示例 1: 接收任意关键字参数
# ============================================================
print("=== 基础示例 1: 接收任意关键字参数 ===")

def print_info(**kwargs):
    """
    打印所有传入的关键字参数
    kwargs 在函数内部是一个字典
    """
    print(f"kwargs 的类型：{type(kwargs)}")
    print(f"kwargs 的内容：{kwargs}")
    print("逐个打印：")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

# 调用时可以传入任意数量的关键字参数
print_info(name="Alice", age=25, city="Beijing")
print()
print_info(job="Engineer", salary=50000)
print()

# ============================================================
# 基础示例 2: 访问特定的关键字参数
# ============================================================
print("=== 基础示例 2: 访问特定的关键字参数 ===")

def create_user(**user_info):
    """创建用户信息，只关心特定的字段"""
    user = {}
    
    # 使用 .get() 安全地访问，提供默认值
    user['name'] = user_info.get('name', 'Unknown')
    user['age'] = user_info.get('age', 0)
    user['email'] = user_info.get('email', 'no-email @example.com')
    
    # 其他额外的信息也保存
    user['extra'] = {k: v for k, v in user_info.items() 
                     if k not in ['name', 'age', 'email']}
    
    return user

user1 = create_user(name="张三", age=30, email="zhangsan @example.com")
print(f"user1: {user1}")

user2 = create_user(name="李四", hobby="reading", city="Shanghai")
print(f"user2: {user2}")
print()

# ============================================================
# 对比：*args vs **kwargs
# ============================================================
print("=== 对比：*args vs **kwargs ===")

def demo_args_kwargs(*args, **kwargs):
    """
    *args  -> 接收位置参数，收集为元组 (tuple)
    **kwargs -> 接收关键字参数，收集为字典 (dict)
    """
    print(f"*args   -> {args}   (类型：{type(args)})")
    print(f"**kwargs -> {kwargs} (类型：{type(kwargs)})")

demo_args_kwargs(1, 2, 3, name="Alice", age=25)
print()

# ============================================================
# 实际应用场景 1: 构建字典/配置
# ============================================================
print("=== 实际应用场景 1: 构建配置对象 ===")

def build_config(**settings):
    """构建配置字典，可以传入任意配置项"""
    default_config = {
        'debug': False,
        'timeout': 30,
        'retry': 3
    }
    
    # 更新默认配置
    default_config.update(settings)
    
    return default_config

config = build_config(debug=True, timeout=60, log_level="INFO")
print(f"配置：{config}")
print()

# ============================================================
# 实际应用场景 2: 转发参数
# ============================================================
print("=== 实际应用场景 2: 转发参数 ===")

def wrapper_function(**kwargs):
    """包装函数，将参数转发给另一个函数"""
    print("wrapper_function 收到参数:", kwargs)
    # 使用 **kwargs 解包字典，作为关键字参数传入
    return original_function(**kwargs)

def original_function(name, age, city="Beijing"):
    """原始函数"""
    print(f"original_function: {name}, {age}岁，来自{city}")

# 调用包装函数
wrapper_function(name="王五", age=28)
wrapper_function(name="赵六", age=35, city="Guangzhou")
print()

# ============================================================
# 实际应用场景 3: 灵活的日志记录
# ============================================================
print("=== 实际应用场景 3: 灵活的日志记录 ===")

def log_message(level, **context):
    """
    记录日志，可以附加任意上下文信息
    """
    timestamp = "2024-01-01 12:00:00"
    print(f"[{timestamp}] [{level}] 日志内容:")
    for key, value in context.items():
        print(f"  - {key}: {value}")

log_message("INFO", user="admin", action="login", ip="192.168.1.1")
print()
log_message("ERROR", error_code=500, message="Server error", module="api")
print()

# ============================================================
# 实际应用场景 4: 数据验证和过滤
# ============================================================
print("=== 实际应用场景 4: 数据验证和过滤 ===")

def validate_data(**data):
    """验证传入的数据"""
    errors = []
    
    # 验证 name
    if 'name' in data and len(data['name']) < 2:
        errors.append("name 至少需要 2 个字符")
    
    # 验证 age
    if 'age' in data and (not isinstance(data['age'], int) or data['age'] < 0):
        errors.append("age 必须是正整数")
    
    # 验证 email
    if 'email' in data and '@' not in data['email']:
        errors.append("email 格式不正确")
    
    if errors:
        return {"valid": False, "errors": errors}
    return {"valid": True, "message": "验证通过"}

result1 = validate_data(name="A", age=-5, email="invalid")
print(f"result1: {result1}")

result2 = validate_data(name="张三", age=25, email="zhangsan @example.com")
print(f"result2: {result2}")
print()

# ============================================================
# 函数参数的完整顺序
# ============================================================
print("=== 函数参数的完整顺序 ===")

def full_example(positional, default="default", *args, keyword_only, **kwargs):
    """
    函数参数定义顺序:
    1. 位置参数 (positional)
    2. 默认参数 (default)
    3. *args (可变位置参数)
    4. 关键字_only 参数 (keyword_only)
    5. **kwargs (可变关键字参数)
    """
    print(f"positional:   {positional}")
    print(f"default:      {default}")
    print(f"*args:        {args}")
    print(f"keyword_only: {keyword_only}")
    print(f"**kwargs:     {kwargs}")

# 调用示例
full_example("必须的值", "自定义默认值", 1, 2, 3, keyword_only="关键字参数", extra="额外信息")
print()

# ============================================================
# 小测验：猜猜输出是什么？
# ============================================================
print("=== 小测验 ===")

def quiz(**data):
    print(f"接收到的参数：{data}")
    print(f"data.get('a', '默认值'): {data.get('a', '默认值')}")
    print(f"'b' in data: {'b' in data}")

quiz(a=1, c=3)
print()

print("恭喜你学完了 **kwargs 的知识！")
