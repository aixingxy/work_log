记录一下使用pycharm调用远程服务器编译环境报ImportError的解决过程：
```text
ssh://xx@192.168.1.xxx:22/usr/bin/python3 -u /home/xxx/pa/preprocess_tacotron.py
Traceback (most recent call last):
  File "/home//.local/lib/python3.5/site-packages/tensorflow/python/pywrap_tensorflow.py", line 58, in <module>
    from tensorflow.python.pywrap_tensorflow_internal import *
  File "/home/xx/.local/lib/python3.5/site-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 28, in <module>
    _pywrap_tensorflow_internal = swig_import_helper()
  File "/home/xx/.local/lib/python3.5/site-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 24, in swig_import_helper
    _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
  File "/usr/lib/python3.5/imp.py", line 242, in load_module
    return load_dynamic(name, filename, file)
  File "/usr/lib/python3.5/imp.py", line 342, in load_dynamic
    return _load(spec)
ImportError: libcublas.so.8.0: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/xx/pa/preprocess_tacotron.py", line 5, in <module>
    from datasets import preprocessor_tacotron as preprocessor
  File "/home/xx/pa/datasets/__init__.py", line 1, in <module>
    from datasets import ljspeech
  File "/home/xx/pa/datasets/ljspeech.py", line 5, in <module>
    import audio
  File "/home/xx/pa/audio.py", line 6, in <module>
    from hparams import hparams
  File "/home/xx/pa/hparams.py", line 1, in <module>
    import tensorflow as tf
  File "/home/xx/.local/lib/python3.5/site-packages/tensorflow/__init__.py", line 24, in <module>
    from tensorflow.python import *
  File "/home/xx/.local/lib/python3.5/site-packages/tensorflow/python/__init__.py", line 49, in <module>
    from tensorflow.python import pywrap_tensorflow
  File "/home/xx/.local/lib/python3.5/site-packages/tensorflow/python/pywrap_tensorflow.py", line 72, in <module>
    raise ImportError(msg)
ImportError: Traceback (most recent call last):
  File "/home/xx/.local/lib/python3.5/site-packages/tensorflow/python/pywrap_tensorflow.py", line 58, in <module>
    from tensorflow.python.pywrap_tensorflow_internal import *
  File "/home/xx/.local/lib/python3.5/site-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 28, in <module>
    _pywrap_tensorflow_internal = swig_import_helper()
  File "/home/xx/.local/lib/python3.5/site-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 24, in swig_import_helper
    _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
  File "/usr/lib/python3.5/imp.py", line 242, in load_module
    return load_dynamic(name, filename, file)
  File "/usr/lib/python3.5/imp.py", line 342, in load_dynamic
    return _load(spec)
ImportError: libcublas.so.8.0: cannot open shared object file: No such file or directory


Failed to load the native TensorFlow runtime.

See https://www.tensorflow.org/install/install_sources#common_installation_problems

for some common reasons and solutions.  Include the entire stack trace
above this error message when asking for help.

Process finished with exit code 1
```

https://www.cnblogs.com/jinggege/p/5766146.html
安装完cuda没有添加lib库路径
```shell
sudo vim /etc/ld.so.conf.d/cuda.conf

# 添加
/usr/local/cuda/lib64
/lib
/usr/lib
/usr/lib32

sudo ldconfig -v
```
