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

