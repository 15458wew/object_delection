


本文为object_delection相关教程

由于校园网网速较慢，代码可参考https://github.com/tensorflow/models/tree/master/research/object_detection

下载TensorFlow Models

1.下载模型

    git clone https://github.com/tensorflow/models.git
  
编译protobuf

    protoc object_detection/protos/*.proto --python_out=.
    生成若干py文件在object_detection/protos/。

添加系统路径

    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

测试

    python object_detection/builders/model_builder_test.py

若成功，显示OK。

2.准备数据

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
        
3.下载模型

官方提供了不少预训练模型（ https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md ），这里以ssd_mobilenet_v1_coco以例。

        #From tensorflow/models/object_detection/
        mkdir checkpoints
        cd checkpoints
        wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
        tar zxf ssd_mobilenet_v1_coco_11_06_2017.tar.gz
        
训练

1.配置

参考 https://github.com/tensorflow/models/blob/master/object_detection/g3doc/configuring_jobs.md
这里使用SSD with MobileNet，把object_detection/samples/configs/ssd_mobilenet_v1_pets.config复制到object_detection/VOC2012/ssd_mobilenet/ssd_mobilenet_v1.config

        #From tensorflow/models/object_detection/VOC2012
        mkdir ssd_mobilenet
        cp ../samples/configs/ssd_mobilenet_v1_pets.config ssd_mobilenet/ssd_mobilenet_v1.config
        
并进行相应的修改：
修改第9行为num_classes: 20。
修改第158行为fine_tune_checkpoint: "../../checkpoints/ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"
修改第177行为input_path: "../data/pascal_train.record"
修改第179行和193行为label_map_path: "../data/pascal_label_map.pbtxt"
修改第191行为input_path: "../data/pascal_val.record"

2.训练

新建object_detection/VOC2012/ssd_mobilenet/train.sh，内容以下：

        mkdir -p logs/
        now=$(date +"%Y%m%d_%H%M%S")
        python ../../train.py \
            --logtostderr \
            --pipeline_config_path=ssd_mobilenet_v1.config \
            --train_dir=train_logs 2>&1 | tee logs/train_$now.txt &
            
进入object_detection/VOC2012/ssd_mobilenet/，运行./train.sh即可训练。

3.验证

可一边训练一边验证，注意使用其它的GPU或合理分配显存。
新建object_detection/VOC2012/ssd_mobilenet/eval.sh，内容以下：

        mkdir -p eval_logs
        python ../../eval.py \
            --logtostderr \
            --pipeline_config_path=ssd_mobilenet_v1.config \
            --checkpoint_dir=train_logs \
            --eval_dir=eval_logs &
            
进入object_detection/VOC2012/ssd_mobilenet/，运行CUDA_VISIBLE_DEVICES="1" ./eval.sh即可验证（这里指定了第二个GPU）。

4.可视化log

可一边训练一边可视化训练的log，访问http://localhost:6006/即可看到Loss等的变化。

        #From tensorflow/models/object_detection/VOC2012/ssd_mobilenet/
        tensorboard --logdir train_logs/
        
可视化验证的log，可看到Precision/mAP@0.5IOU的变化以及具体image的预测结果，这里指定了另一个端口。

        #From tensorflow/models/object_detection/VOC2012/ssd_mobilenet/
        tensorboard --logdir eval_logs/ --port 6007

或同时可视化训练与验证的log：

        #From tensorflow/models/object_detection/VOC2012/ssd_mobilenet/
        tensorboard --logdir .

