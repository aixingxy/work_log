# TensorFlow固化模型+打包程序
## 训练过程保存模型
Tensorflow在训练过程中将参数和graph分开保存，例如使用下面的代码：
``` python
# -*- coding:utf-8 -*-
import tensorflow as tf
import os

dir = os.path.dirname(os.path.realpath(__file__))

v1 = tf.Variable(1, name='v1')
v2 = tf.placeholder(tf.int32, name='v2')

y = tf.add(v1, v2, name='add')

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(y, feed_dict={v2: 2}))

    save_dir = dir+'/model'
    os.makedirs(save_dir, exist_ok=True)
    saver.save(sess, save_dir+'/model')
```
会生成4个文件，当然在训练的过程中除了checkpoint，其他三个文件会有多个。
``` text
checkpoint
model.data-00000-of-00001
model.index
model.meta
```

简单描述几个文件：
meta文件是保存图的（包括图，操作等)
data文件是保存数据的（权重）
index文件是一个不可修改的键值表
## 固化训练好的模型
在训练完成后选择效果最好的模型，进行压缩，或者将graph和权重放在一起以便生产使用。
``` python
# -*- coding:utf-8 -*-
import tensorflow as tf
import os

dir = os.path.dirname(os.path.realpath(__file__))
checkpoint = tf.train.get_checkpoint_state(dir + '/model')
input_checkpoint = checkpoint.model_checkpoint_path
print(input_checkpoint)

absolute_model = '/'.join(input_checkpoint.split('/')[:-1])
print(absolute_model)

output_grap = absolute_model + "/frozen_model.pb"
with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)

    saver.restore(sess, input_checkpoint)
    # 打印图中的变量，查看要保存的
    for op in tf.get_default_graph().get_operations():
        print(op.name, op.values())

    output_grap_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                   tf.get_default_graph().as_graph_def(),
                                                                   output_node_names=['add'])
    with tf.gfile.GFile(output_grap, 'wb') as f:
        f.write(output_grap_def.SerializeToString())
    print("%d ops in the final graph." % len(output_grap_def.node))
```
此时model文件夹下就会多出frozen_model.pb文件
convert_variables_to_constants()
1. 会将变量替换成常量固化起来
2. 将前向传播不需要的节点node去掉
所以output_node_names参数只要输入你的网络的输出，就会生成一个最小的序列化的二进制pb文件。

## 使用pb(protobuf)模型
``` python
# -*- coding:utf-8 -*-
import tensorflow as tf
import argparse
def load_graph(frozen_graph_file):
    with tf.gfile.GFile(frozen_graph_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='prefix')
    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default='frozen_model.pb',
                        type=str, help='Frozen model file to import')
    args = parser.parse_args()

    graph = load_graph(args.frozen_model_filename)

    v2 = graph.get_tensor_by_name('prefix/v2:0')
    add = graph.get_tensor_by_name('prefix/add:0')

    for op in graph.get_operations():
        print(op.name)

    with tf.Session(graph=graph) as sess:

        out = sess.run(add, feed_dict={v2: 10})
        print(out)
```

## 打包程序
上面的模型已经打包了，下面对test.py代码进行打包，与上面的不同的地方是将加法的第二个参数预留出来
``` python
# -*- coding:utf-8 -*-
import os
os.environ["PBR_VERSION"]='3.1.1'
import argparse
import tensorflow as tf



def load_graph(frozen_graph_file):
    with tf.gfile.GFile(frozen_graph_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='prefix')
    return graph


if __name__ == "__main__":
    # 创建一个解析对象
    parser = argparse.ArgumentParser()
    # 向parser对象中添加命令行参数和选项参数
    parser.add_argument('--num', type=int, help='add') # 留出加法的第二个数子作为参数
    parser.add_argument("--frozen_model_filename",
                        default='model/frozen_model.pb',
                        type=str, help='Frozen model file to import')
    # 进行解析
    args = parser.parse_args()

    graph = load_graph(args.frozen_model_filename)
    v2 = graph.get_tensor_by_name('prefix/v2:0')
    add = graph.get_tensor_by_name('prefix/add:0')

    with tf.Session(graph=graph) as sess:
        out = sess.run(add, feed_dict={v2: args.num})
        print(out)
```
使用 `python test.py --num=10`
输出 11

``` bash
# 安装pyinstaller
# pip install pyinstaller
# -F 是 --onefile的缩写
# --clean 是清理临时文件，也就是build文件夹下的临时文件
pyinstaller -F  --clean test.py
```
完成后到dist文件夹下
``` bash
./test --num=10
```

如果没有os.environ["PBR_VERSION"]='3.1.1'会报错
``` text
Traceback (most recent call last):
  File "pack_tf_add.py", line 4, in <module>
  File "/private/var/folders/88/1jw_0lt50tsb4n08mg_493040000gn/T/pip-build-3m08rf/pyinstaller/PyInstaller/loader/pyimod03_importers.py", line 396, in load_module
  File "site-packages/tensorflow/__init__.py", line 24, in <module>
  File "/private/var/folders/88/1jw_0lt50tsb4n08mg_493040000gn/T/pip-build-3m08rf/pyinstaller/PyInstaller/loader/pyimod03_importers.py", line 396, in load_module
  File "site-packages/tensorflow/python/__init__.py", line 104, in <module>
  File "/private/var/folders/88/1jw_0lt50tsb4n08mg_493040000gn/T/pip-build-3m08rf/pyinstaller/PyInstaller/loader/pyimod03_importers.py", line 396, in load_module
  File "site-packages/tensorflow/python/platform/test.py", line 53, in <module>
  File "/private/var/folders/88/1jw_0lt50tsb4n08mg_493040000gn/T/pip-build-3m08rf/pyinstaller/PyInstaller/loader/pyimod03_importers.py", line 396, in load_module
  File "site-packages/mock/__init__.py", line 2, in <module>
  File "/private/var/folders/88/1jw_0lt50tsb4n08mg_493040000gn/T/pip-build-3m08rf/pyinstaller/PyInstaller/loader/pyimod03_importers.py", line 396, in load_module
  File "site-packages/mock/mock.py", line 71, in <module>
  File "site-packages/pbr/version.py", line 462, in semantic_version
  File "site-packages/pbr/version.py", line 449, in _get_version_from_pkg_resources
  File "site-packages/pbr/packaging.py", line 812, in get_version
Exception: Versioning for this project requires either an sdist tarball, or access to an upstream git repository. It's also possible that there is a mismatch between the package name in setup.cfg and the argument given to pbr.version.VersionInfo. Project name mock was given, but was not able to be found.
```
解决方法：

https://blog.csdn.net/laocaibcc229/article/details/78570017
https://github.com/pyinstaller/pyinstaller/issues/2883
``` python
# 添加到首行
import os
os.environ["PBR_VERSION"]='3.1.1' #要去查询自己的版本
```

查看pbr版本
``` bash
pbr --version # 3.1.1
```
## web API
使用flask搭建一个微型web
``` python
# -*- coding:utf-8 -*-
import argparse
from flask import Flask
from flask import request
import tensorflow as tf

app = Flask(__name__)


def load_graph(frozen_graph_file):
    with tf.gfile.GFile(frozen_graph_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='prefix')
    return graph


@app.route('/', methods=['POST', 'GET'])
def about():
    if request.method == "POST":
        print("in post")
        num = request.form.get('num')
        y_out = persistent_sess.run(y, feed_dict={x: num})

        return str(y_out)
    else:
        return """<form action="/" method="POST">
                  <input type="text" name="num" placeholder="Enter num">
                  <input type="submit" value="Submit" name="ok"/>
                  </form>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="frozen_model.pb", type=str,
                        help="Frozen model file to import")
    parser.add_argument("--gpu_memory", default=.2, type=float, help="GPU memory per process")
    args = parser.parse_args()

    print('Loading the model')
    graph = load_graph(args.frozen_model_filename)
    x = graph.get_tensor_by_name('prefix/v2:0')
    y = graph.get_tensor_by_name('prefix/add:0')
    # use gpu
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    # sess_config = tf.ConfigProto(gpu_options=gpu_options)
    # persistent_sess = tf.Session(graph=graph, config=sess_config)

    # use cpu
    persistent_sess = tf.Session(graph=graph)
    print('Starting the API')
    app.run()
```
点击 http://127.0.0.1:5000/ 输入数字

点击 submit 显示结果
