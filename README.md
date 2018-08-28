

本文为object_delection相关教程

由于校园网网速较慢，代码可参考https://github.com/tensorflow/models/tree/master/research/object_detection

下载TensorFlow Models

下载模型

    git clone https://github.com/tensorflow/models.git
  
编译protobuf

    protoc object_detection/protos/*.proto --python_out=.
    生成若干py文件在object_detection/protos/。

添加系统路径

    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

测试

    python object_detection/builders/model_builder_test.py

若成功，显示OK。

准备数据

参考 https://github.com/tensorflow/models/blob/master/object_detection/g3doc/preparing_inputs.md
这里以PASCAL VOC 2012为例。
下载并解压

        #From tensorflow/models/object_detection
        mkdir -p VOC2012/data
        cd VOC2012/data
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
        tar -xvf VOCtrainval_11-May-2012.tar
生成TFRecord

        #From tensorflow/models/object_detection/VOC2012/
        cp ../../create_pascal_tf_record.py .
        cp ../../data/pascal_label_map.pbtxt data/
        python create_pascal_tf_record.py \
            --label_map_path=data/pascal_label_map.pbtxt \
            --data_dir=data/VOCdevkit --year=VOC2012 --set=train \
            --output_path=data/pascal_train.record
        python create_pascal_tf_record.py \
            --label_map_path=data/pascal_label_map.pbtxt \
            --data_dir=data/VOCdevkit --year=VOC2012 --set=val \
            --output_path=data/pascal_val.record

得到data/pascal_train.record和data/pascal_val.record。

如果需要用自己的数据，则参考create_pascal_tf_record.py编写处理数据生成TFRecord的脚本。可参考

        https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md
