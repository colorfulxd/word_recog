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
