import eval_image_classifier as eic

eic.FLAGS.dataset_name= 'quiz'
eic.FLAGS.dataset_dir= 'C:/Work/06_Learn/Class11/Data/train'
eic.FLAGS.dataset_split_name= 'validation'
eic.FLAGS.checkpoint_path= 'C:/Work/06_Learn/Class11/Data/TinyMindTrain'
eic.FLAGS.model_name= 'inception_v3'
eic.FLAGS.eval_dir= 'C:/Work/06_Learn/Class11/Data/train/eval'
eic.FLAGS.max_num_batches= 24

eic.FLAGS.batch_size= 2 #不能是1报错
eic.main("")

#python C:/Work/06_Learn/Class11/quiz-word-recog/train_eval_image_classifier.py --learning_rate 0.001 --batch_size 32 --output_dir C:/Work/06_Learn/Class11/train/output --dataset_name quiz --dataset_dir C:/Work/06_Learn/Class11/train --checkpoint_path C:/Work/06_Learn/Class11/inception_v3.ckpt --model_name inception_v3 --checkpoint_exclude_scopes InceptionV3/Logits,InceptionV3/AuxLogits --train_dir C:/Work/06_Learn/Class11/train --dataset_split_name train --eval_dir C:/Work/06_Learn/Class11/train/output/eval --max_num_batches 128 --clone_on_cpu True --optimizer rmsprop
#训练
#python C:/Work/06_Learn/Class11/quiz-word-recog/train_image_classifier.py --learning_rate 0.005 --batch_size 2 --output_dir C:/Work/06_Learn/Class11/train/output --dataset_name quiz --dataset_dir C:/Work/06_Learn/Class11/train --checkpoint_path C:/Work/06_Learn/Class11/inception_v3.ckpt --model_name inception_v3 --checkpoint_exclude_scopes InceptionV3/Logits,InceptionV3/AuxLogits --train_dir C:/Work/06_Learn/Class11/train --dataset_split_name train --eval_dir C:/Work/06_Learn/Class11/train/output/eval --max_num_batches 128 --clone_on_cpu True --optimizer rmsprop

#验证
#python C:/Work/06_Learn/Class11/quiz-word-recog/eval_image_classifier.py --batch_size 2 --output_dir C:/Work/06_Learn/Class11/train/output --dataset_name quiz --dataset_dir C:/Work/06_Learn/Class11/train --checkpoint_path C:/Work/06_Learn/Class11/train --model_name inception_v3 --dataset_split_name validation --eval_dir C:/Work/06_Learn/Class11/train/output/eval --max_num_batches 128 --clone_on_cpu True

#导出和freeze模型
#sh export_and_freeze.sh
#sh classify_image.sh
#sh server.sh

#训练
#python C:/Work/06_Learn/Class11/quiz-word-recog/train_image_classifier.py --learning_rate 0.005 --batch_size 2 --output_dir C:/Work/06_Learn/Class11/train/OutputDense --dataset_name quiz --dataset_dir C:/Work/06_Learn/Class11/train --model_name densenet --train_dir C:/Work/06_Learn/Class11/train/trainDense --dataset_split_name train --eval_dir C:/Work/06_Learn/Class11/train/output/evalDense --max_num_batches 128 --clone_on_cpu True --optimizer rmsprop

#验证
#python C:/Work/06_Learn/Class11/quiz-word-recog/eval_image_classifier.py --batch_size 2 --output_dir C:/Work/06_Learn/Class11/train/output1 --dataset_name quiz --dataset_dir C:/Work/06_Learn/Class11/train --checkpoint_path C:/Work/06_Learn/Class11/train/output1 --model_name densenet --dataset_split_name validation --eval_dir C:/Work/06_Learn/Class11/train/output/evalDense --max_num_batches 128 --clone_on_cpu True
