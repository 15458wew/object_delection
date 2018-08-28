


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
        
测试

1.导出模型

训练完成后得到一些checkpoint文件在tensorflow/models/object_detection/VOC2012/ssd_mobilenet/train_logs/中，如：

graph.pbtxt

model.ckpt-200000.data-00000-of-00001

model.ckpt-200000.info

model.ckpt-200000.meta

其中meta保存了graph和metadata，ckpt保存了网络的weights。

而进行预测时只需模型和权重，不需要metadata，故可使用官方提供的脚本生成推导图。

        #From tensorflow/models/object_detection/VOC2012/ssd_mobilenet/
        mkdir -p output
        CUDA_VISIBLE_DEVICES="1" python ../../export_inference_graph.py \
            --input_type image_tensor \
            --pipeline_config_path ssd_mobilenet_v1.config \
            --trained_checkpoint_prefix train_logs/model.ckpt-200000 \
            --output_directory output/
            
2.测试图片
运行object_detection_tutorial.ipynb并修改其中的各种路径即可。

或自写inference脚本，如tensorflow/models/object_detection/VOC2012/infer.py

        import sys
        sys.path.append('..')
        import os
        import time
        import tensorflow as tf
        import numpy as np
        from PIL import Image
        from matplotlib import pyplot as plt

        from utils import label_map_util
        from utils import visualization_utils as vis_util

        if len(sys.argv) < 3:
            print('Usage: python {} test_image_path checkpoint_path'.format(sys.argv[0]))
            exit()

        PATH_TEST_IMAGE = sys.argv[1]
        PATH_TO_CKPT = sys.argv[2]
        PATH_TO_LABELS = 'data/pascal_label_map.pbtxt'
        NUM_CLASSES = 21
        IMAGE_SIZE = (18, 12)

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph, config=config) as sess:
                start_time = time.time()
                print(time.ctime())
                image = Image.open(PATH_TEST_IMAGE)
                image_np = np.array(image).astype(np.uint8)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                print('{} elapsed time: {:.3f}s'.format(time.ctime(), time.time() - start_time))
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                    category_index, use_normalized_coordinates=True, line_thickness=8)
                plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(image_np)

