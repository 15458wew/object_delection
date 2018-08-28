本文为object_delection相关教程

由于校园网网速较慢，代码可参考https://github.com/tensorflow/models/tree/master/research/object_detection

下载TensorFlow Models

下载模型

  git clone https://github.com/tensorflow/models.git
  
编译protobuf
  # From tensorflow/models/
  protoc object_detection/protos/*.proto --python_out=.
  生成若干py文件在object_detection/protos/。

添加系统路径
  # From tensorflow/models/
  export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

测试
  # From tensorflow/models/
  python object_detection/builders/model_builder_test.py

若成功，显示OK。
