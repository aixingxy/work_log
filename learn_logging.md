# logging模块的学习
可以在一个py文件中定义一个logger对象，在需要记录日志的程序中导入这个对象，就可以在一个程序中将所有日志写入一个文件或同时输出到控制台
```python
# /usr/bin/python3
# -*- coding:utf-8 -*- import logging
# log.py

logger = logging.getLogger()

fh = logging.FileHandler(filename='logger.log', mode='w')  # 文件输出
# filename='logger.log'  # 设置输出文件名
# mode='w'  # 设置写入方法
ch = logging.StreamHandler()  # 控制台输出

fm = logging.Formatter('%(asctime)s %(filename)s %(lineno)d %(message)s')  # 设置输出格式
fh.setFormatter(fm)
ch.setFormatter(fm)

logger.addHandler(fh)  
logger.addHandler(ch)

logger.setLevel(logging.DEBUG)  # 设置显示等级
# debug->info->warning->error->critical设置哪一个就显示包含这个等级及以上的等级的信息
```

在需要记录日志的地方导入log.py文件中创建的logger对象
```python
# test1.py
from log import logger
logger.debug("this is dubug in test1")
```

```python
# test2.py
from log import logger
logger.debug("this is dubug in test2")
```
这样就可以记录一个项目中的所有日志信息了

## logging.Formatter常用格式说明
|||
|-|-|
|%(name)s|Logger的名字|
|%(levelno)s|数字形式的日志级别|
|%(levelname)s|文本形式的日志级别|
|%(pathname)s|调用日志输出函数模块的`完整路径`，可能没有|
|%(filename)s|调用日志输出函数模块的`文件名`|
|%(module)s|调用日志输出函数的`模块名`|
|%(funcName)s|调用日志输出函数的`函数名`|
|%(lineno)s|调用日志输出函数`语句所在行`|
|%(created)s|当前时间，用UNIX标准的表示时间的浮点数表示|
|%(relativeCreated)s|输出日志信息时的，自Logger创建以来的毫秒数 |
|%(asctime)s|`字符串形式的当前时间`。默认格式是“2003-07-08 16:49:45,896”。逗号后面的是毫秒 |
|%(thread)s|线程ID。可能没有 |
|%(threadName)s|线程名。可能没有 |
|%(process)s|进程ID。可能没有 |
|%(message)s|`用户输出的消息` |


  程序员可以使用三种方式配置 logging：

**1.使用 Python 代码调用前面提到的配置类方法来显式地创建 loggers，handlers，formatters** 

**2.创建一个 logging 的配置文件，然后使用 fileConfig() 来读取该文件**

**3.创建一个保存了配置信息的字典，然后把它传递给 dictConfig() 函数**

下面的示例中，使用 Python 代码配置了一个非常简单的 logger，一个控制台处理器，和一个简单的格式化器：
```python
import logging

# create logger logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch ch.setFormatter(formatter)

# add ch to logger logger.addHandler(ch)

# 'application' code logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')
```
从命令行运行上面的代码，会产生如下输出：
```text
2018-08-12 13:48:03,982 - simple_example - DEBUG - debug message
2018-08-12 13:48:03,982 - simple_example - INFO - info message
2018-08-12 13:48:03,982 - simple_example - WARNING - warn message
2018-08-12 13:48:03,982 - simple_example - ERROR - error message
2018-08-12 13:48:03,982 - simple_example - CRITICAL - critical message
```
