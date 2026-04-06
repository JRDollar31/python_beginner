# 07_classes.py
# Python 类

# 1. 定义一个简单的类
print("=== 定义一个简单的类 ===")

class Dog:
    """一个简单的狗类"""
    
    def __init__(self, name, age):
        """初始化方法"""
        self.name = name
        self.age = age
    
    def bark(self):
        """狗叫的方法"""
        print(f"{self.name} 在叫：汪汪汪！")
    
    def get_info(self):
        """获取狗的信息"""
        return f"{self.name}, {self.age}岁"

# 创建对象
my_dog = Dog("旺财", 3)
print(f"名字: {my_dog.name}")
print(f"年龄: {my_dog.age}")
my_dog.bark()
print(my_dog.get_info())

# 2. 类的继承
print("\n=== 类的继承 ===")

class Animal:
    """动物基类"""
    
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        print("动物在说话")

class Cat(Animal):
    """猫类，继承自动物类"""
    
    def speak(self):
        print(f"{self.name} 在叫：喵喵喵！")

class Bird(Animal):
    """鸟类，继承自动物类"""
    
    def speak(self):
        print(f"{self.name} 在叫：叽叽叽！")

cat = Cat("咪咪")
cat.speak()

bird = Bird("小鸟")
bird.speak()

# 3. 类变量和实例变量
print("\n=== 类变量和实例变量 ===")

class Counter:
    """计数器类"""
    
    count = 0  # 类变量
    
    def __init__(self, name):
        self.name = name  # 实例变量
        Counter.count += 1
    
    def get_count(self):
        return Counter.count

c1 = Counter("计数器 1")
c2 = Counter("计数器 2")
c3 = Counter("计数器 3")

print(f"创建的实例数量: {Counter.count}")
print(f"c1 的名字: {c1.name}")
print(f"c2 的名字: {c2.name}")

# 4. 私有属性
print("\n=== 私有属性 ===")

class Person:
    """人类"""
    
    def __init__(self, name, age):
        self.name = name
        self.__age = age  # 私有属性
    
    def get_age(self):
        """获取年龄"""
        return self.__age
    
    def set_age(self, age):
        """设置年龄"""
        if age > 0:
            self.__age = age

person = Person("Alice", 25)
print(f"姓名: {person.name}")
print(f"年龄 (通过方法访问): {person.get_age()}")
person.set_age(26)
print(f"修改后的年龄: {person.get_age()}")

# 5. 静态方法
print("\n=== 静态方法 ===")

class MathUtils:
    """数学工具类"""
    
    @staticmethod
    def add(a, b):
        return a + b
    
    @staticmethod
    def multiply(a, b):
        return a * b

print(f"5 + 3 = {MathUtils.add(5, 3)}")
print(f"4 * 6 = {MathUtils.multiply(4, 6)}")

# 6. 类方法
print("\n=== 类方法 ===")

class Student:
    """学生类"""
    
    school = "第一中学"
    
    def __init__(self, name):
        self.name = name
    
    @classmethod
    def get_school(cls):
        return cls.school
    
    @classmethod
    def from_string(cls, student_str):
        name = student_str.split("-")[0]
        return cls(name)

print(f"学校: {Student.get_school()}")
student = Student.from_string("小明-15")
print(f"学生姓名: {student.name}")

# 7. 特殊方法（魔术方法）
print("\n=== 特殊方法 ===")

class Book:
    """书籍类"""
    
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
    
    def __str__(self):
        return f"《{self.title}》by {self.author}"
    
    def __len__(self):
        return self.pages
    
    def __eq__(self, other):
        return self.title == other.title and self.author == other.author

book1 = Book("Python 入门", "张三", 200)
book2 = Book("Python 入门", "张三", 200)

print(f"book1: {book1}")
print(f"book1 的页数: {len(book1)}")
print(f"book1 == book2: {book1 == book2}")
