import train_image_classifier as tic

tic.FLAGS.dataset_name= 'quiz'
tic.FLAGS.dataset_dir= 'C:/Work/06_Learn/Class11/Data/train'

tic.FLAGS.checkpoint_path= 'C:/Work/06_Learn/Class11/inception_v3.ckpt'
tic.FLAGS.model_name= 'inception_v3'
tic.FLAGS.checkpoint_exclude_scopes= 'InceptionV3/Logits,InceptionV3/AuxLogits'

tic.FLAGS.train_dir= 'C:/Work/06_Learn/Class11/Data/train'
tic.FLAGS.learning_rate= 0.001
tic.FLAGS.optimizer= 'rmsprop'

tic.FLAGS.batch_size= 1
tic.main("")

# python3 C:/Work/06_Learn/Class11/quiz-word-recog/train_image_classifier.py -
# -dataset_name= quiz --dataset_dir = C:/Work/06_Learn/Class11/Data/train --checkp
# oint_path=C:/Work/06_Learn/Class11//inception_v3.ckpt --model_name = inception_v
# 3 --checkpoint_exclude_scopes= InceptionV3/Logits,InceptionV3/AuxLogits --train_
# dir = C:/Work/06_Learn/Class11/Data/train -- learning_rate =0.001 --optimizer =
# rmsprop --batch_size=32

