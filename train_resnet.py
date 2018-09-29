"""
Example TensorFlow script for finetuning a VGG model on your own data.
Uses tf.contrib.data module which is in release v1.2
Based on PyTorch example from Justin Johnson
(https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c)
Required packages: tensorflow (v1.2)
Download the weights trained on ImageNet for VGG:
```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
rm vgg_16_2016_08_28.tar.gz
```
For this example we will use a tiny dataset of images from the COCO dataset.
We have chosen eight types of animals (bear, bird, cat, dog, giraffe, horse,
sheep, and zebra); for each of these categories we have selected 100 training
images and 25 validation images from the COCO dataset. You can download and
unpack the data (176 MB) by running:
```
wget cs231n.stanford.edu/coco-animals.zip
unzip coco-animals.zip
rm coco-animals.zip
```
The training data is stored on disk; each category has its own folder on disk
and the images for that category are stored as .jpg files in the category folder.
In other words, the directory structure looks something like this:
coco-animals/
  train/
    bear/
      COCO_train2014_000000005785.jpg
      COCO_train2014_000000015870.jpg
      [...]
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
  val/
    bear/
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
"""

import argparse
import os
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import cv2
import tf_data
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='./data/output')
parser.add_argument('--val_dir', default='./test')
parser.add_argument('--model_path', default='resnet_v1_101.ckpt', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=10, type=int)
parser.add_argument('--num_epochs2', default=10, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

#VGG_MEAN = [123.68, 116.78, 103.94]
VGG_MEAN = [0.0]

#================ DATASET INFORMATION ======================
#State dataset directory where the tfrecord files are located
#dataset_dir = '/home/anshuman/mnist_png/new_training/'

#State where your log file is at. If it doesn't exist, create it.
log_dir = './log'

#State where your checkpoint file is
checkpoint_file = './resnet_v1_101.ckpt'

#State the image size you're resizing your images to. We will use the default inception size of 299.
image_size = 480

#State the number of classes to predict:
num_classes = 10

#State the labels file and read it
#labels_file = '/home/anshuman/mnist_png/new_training/labels.txt'
#labels = open(labels_file, 'r')
#================= TRAINING INFORMATION ==================
#State the number of epochs to train
num_epochs = 5

#State your batch size
batch_size = 8

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2

def placeholder_inputs(batch_size):
	"""Generate placeholder variables to represent the input tensors.

	These placeholders are used as inputs by the rest of the model building
	code and will be fed from the downloaded data in the .run() loop, below.

	Args:
	batch_size: The batch size will be baked into both placeholders.

	Returns:
	images_placeholder: Images placeholder.
	labels_placeholder: Labels placeholder.
	"""
	# Note that the shapes of the placeholders match the shapes of the full
	# image and label tensors, except the first dimension is now batch_size
	# rather than the full size of the train or test data sets.
	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
	                                                     480,480,3))
	labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
	return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
	"""Fills the feed_dict for training the given step.

	A feed_dict takes the form of:
	feed_dict = {
	  <placeholder>: <tensor of values to be passed for placeholder>,
	  ....
	}

	Args:
	data_set: The set of images and labels, from input_data.read_data_sets()
	images_pl: The images placeholder, from placeholder_inputs().
	labels_pl: The labels placeholder, from placeholder_inputs().

	Returns:
	feed_dict: The feed dictionary mapping from placeholders to values.
	"""
	# Create the feed_dict for the placeholders filled with the next
	# `batch size ` examples.

	images_feed, labels_feed = data_set.next_batch(batch_size,
	                                             False)
	#print (images_feed.shape)

	feed_dict = {
	  images_pl: images_feed,
	  labels_pl: labels_feed,
	}
	#print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
	#print(labels_feed)
	return feed_dict


def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    labels = os.listdir(directory)
    # Sort the labels so that training and validation get them in the same order
    labels.sort()

    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i

    labels = [label_to_int[l] for l in labels]

    return filenames, labels


def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc


def main(args):
	# Get the list of filenames and corresponding list of labels for training et validation
	#train_filenames, train_labels = list_images(args.train_dir)
	#val_filenames, val_labels = list_images(args.val_dir)
	#assert set(train_labels) == set(val_labels),\
	#       "Train and val labels don't correspond:\n{}\n{}".format(set(train_labels),
	#                                                               set(val_labels))

	#num_classes = len(set(train_labels))
	#print num_classes
	#dataset_size = len(train_labels)
	#print dataset_size
	#print train_filenames
	#print type(train_filenames)
	#exit()
	'''
	x_train = []
	y_train = []
	x_val = []
	y_val = []	

	for file in train_filenames:
		x_train.append(cv2.imread(file))
	for lab in train_labels :
		y_train.append(lab)

	for file in val_filenames:
		x_val.append(cv2.imread(file))
	for lab in val_labels :
		y_val.append(lab)
	#print x_train
	print type(x_train[0])
	#exit()	
	'''

	# --------------------------------------------------------------------------
	# In TensorFlow, you first want to define the computation graph with all the
	# necessary operations: loss, training op, accuracy...
	# Any tensor created in the `graph.as_default()` scope will be part of `graph`
	data_sets = tf_data.read_data_sets(args.train_dir, False)

	graph = tf.Graph()
	with graph.as_default():

		tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level        
		# Standard preprocessing for VGG on ImageNet taken from here:
		# https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
		# Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

		# Preprocessing (for both training and validation):
		# (1) Decode the image from jpg format
		# (2) Resize the image so its smaller side is 256 pixels long
		'''
		def _parse_function(image, label):

			#print type(image)
			#print image
			#im = cv2.imread(filename)
			#print im
			#print type(im)
			#exit()
			print
			image_string = tf.read_file(filename)
			#print image_string
			image_decoded = tf.image.decode_png(image_string, channels=3)          # (1)
			print image_decoded.shape
			#print image_decoded
			#image = np.array(image)
			#image_decoded = tf.convert_to_tensor(image,)
			print image_decoded.shape
			print image_decoded

			image = tf.cast(image_decoded, tf.float32)
			print image.shape
			print image
			#exit()
			smallest_side = 256.0
			height, width = tf.shape(image)[0], tf.shape(image)[1]
			height = tf.to_float(height)
			width = tf.to_float(width)

			scale = tf.cond(tf.greater(height, width),
			                lambda: smallest_side / width,
			                lambda: smallest_side / height)
			new_height = tf.to_int32(height * scale)
			new_width = tf.to_int32(width * scale)

			resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
			return resized_image, label

		#_,_ = _parse_function('./train/0/1.tif',[0])
		#exit()
		# Preprocessing (for training)
		# (3) Take a random 224x224 crop to the scaled image
		# (4) Horizontally flip the image with probability 1/2
		# (5) Substract the per color mean `VGG_MEAN`
		# Note: we don't normalize the data here, as VGG was trained without normalization
		def training_preprocess(image, label):
		    crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
		    flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

		    means = tf.reshape(tf.constant(VGG_MEAN), [1])#, 1, 1])
		    centered_image = flip_image - means                                     # (5)

		    return centered_image, label

		# Preprocessing (for validation)
		# (3) Take a central 224x224 crop to the scaled image
		# (4) Substract the per color mean `VGG_MEAN`
		# Note: we don't normalize the data here, as VGG was trained without normalization
		def val_preprocess(image, label):
		    crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

		    means = tf.reshape(tf.constant(VGG_MEAN), [1])#, 1, 1])
		    centered_image = crop_image - means                                     # (4)

		    return centered_image, label

		# ----------------------------------------------------------------------
		# DATASET CREATION using tf.contrib.data.Dataset
		# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

		# The tf.contrib.data.Dataset framework uses queues in the background to feed in
		# data to the model.
		# We initialize the dataset with a list of filenames and labels, and then apply
		# the preprocessing functions described above.
		# Behind the scenes, queues will load the filenames, preprocess them with multiple
		# threads and apply the preprocessing in parallel, and then batch the data

		#print train_labels
		# Training dataset
		#print(train_filenames)
		train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
		#train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
		train_dataset = train_dataset.map(_parse_function)#,
		     #output_buffer_size=args.batch_size)
		#train_dataset = train_dataset.map(training_preprocess)#,
		     #output_buffer_size=args.batch_size)
		train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
		batched_train_dataset = train_dataset.batch(batch_size)

		# Validation dataset
		val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
		val_dataset = val_dataset.map(_parse_function)
		#val_dataset = val_dataset.map(val_preprocess)#,
		#    num_threads=args.num_workers, output_buffer_size=args.batch_size)
		batched_val_dataset = val_dataset.batch(batch_size)


		# Now we define an iterator that can operator on either dataset.
		# The iterator can be reinitialized by calling:
		#     - sess.run(train_init_op) for 1 epoch on the training set
		#     - sess.run(val_init_op)   for 1 epoch on the valiation set
		# Once this is done, we don't need to feed any value for images and labels
		# as they are automatically pulled out from the iterator queues.

		# A reinitializable iterator is defined by its structure. We could use the
		# `output_types` and `output_shapes` properties of either `train_dataset`
		# or `validation_dataset` here, because they are compatible.
		iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types,
		                                                   batched_train_dataset.output_shapes)
		images, labels = iterator.get_next()
		'''
		images, labels = placeholder_inputs(batch_size)


		#train_init_op = iterator.make_initializer(batched_train_dataset)
		#val_init_op = iterator.make_initializer(batched_val_dataset)

		# Indicates whether we are in training or in test mode
		is_training = tf.placeholder(tf.bool)

		# ---------------------------------------------------------------------
		# Now that we have set up the data, it's time to set up the model.
		# For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
		# last fully connected layer (fc8) and replace it with our own, with an
		# output size num_classes=8
		# We will first train the last layer for a few epochs.
		# Then we will train the entire model on our dataset for a few epochs.

		# Get the pretrained model, specifying the num_classes argument to create a new
		# fully connected replacing the last one, called "vgg_16/fc8"
		# Each model has a different architecture, so "vgg_16/fc8" will change in another model.
		# Here, logits gives us directly the predicted scores we wanted from the images.
		# We pass a scope to initialize "vgg_16/fc8" weights with he_initializer

		#vgg = tf.contrib.slim.nets.vgg
		#Know the number steps to take before decaying the learning rate and batches per epoch
		num_batches_per_epoch = int((data_sets.train._num_examples) / batch_size)
		num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed
		decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

		res = tf.contrib.slim.nets.resnet_v1
		#with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
		#    logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training,
		#                           dropout_keep_prob=args.dropout_keep_prob)
		#print images.shape
		print("shape of labels  "+str(labels.shape)) 
		#print labels
		with slim.arg_scope(res.resnet_arg_scope()):
		    logits, end_points = res.resnet_v1_101(images, num_classes = 10, is_training=True)
		print("shape of logits" +str( logits.shape))
		#print end_points.keys()
		logits = tf.reshape(logits,[-1,10])
		print("shape of logits"+str(logits.shape))        
		#Define the scopes that you want to exclude for restoration
		exclude = ['resnet_v1_101/logits','resnet_v1_101/Predictions']#resnet_v1_50','resnet_v1_152','resnet_v1_200','bottleneck_v1']
		#exclude = []
		variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

		one_hot_labels = slim.one_hot_encoding(labels, 10)#dataset.num_classes)
		#print ("shape of one_hot_labels"+str(one_hot_labels.shape))

		#Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
		loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
		#loss = tf.losses.softmax_cross_entropy(onehot_labels = labels, logits = logits)        
		print ("shape of loss"+str(loss.shape))
		total_loss = tf.losses.get_total_loss()    #obtain the regularization losses as well

		#Create the global step for monitoring the learning_rate and training.
		global_step = get_or_create_global_step()

		#Define your exponentially decaying learning rate
		lr = tf.train.exponential_decay(
		    learning_rate = initial_learning_rate,
		    global_step = global_step,
		    decay_steps = decay_steps,
		    decay_rate = learning_rate_decay_factor,
		    staircase = True)

		#Now we can define the optimizer that takes on the learning rate
		optimizer = tf.train.AdamOptimizer(learning_rate = lr)

		#Create the train_op.
		train_op = slim.learning.create_train_op(total_loss, optimizer)

		print("end point pred shape " + str(end_points['predictions'].shape))
		end_points['predictions'] = tf.reshape(end_points['predictions'],[-1,10])
		#State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
		predictions = tf.argmax(end_points['predictions'], 1)
		#print predictions
		print ("pred shape"+str(predictions.shape))
		print ("labels shape"+str(labels.shape))
		probabilities = end_points['predictions']
		print ("prob shape"+str(probabilities.shape))

		accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
		metrics_op = tf.group(accuracy_update, probabilities)

		#Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
		def train_step(sess, train_op, global_step):
		    '''
		    Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
		    '''
		    #Check the time for each sess run
		    start_time = time.time()
		    total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
		    time_elapsed = time.time() - start_time

		    #Run the logging to print some results
		    logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

		    return total_loss, global_step_count



		# Specify where the model checkpoint is (pretrained weights).
		model_path = args.model_path
		assert(os.path.isfile(model_path))
		#Now finally create all the summaries you need to monitor and group them into one summary op.
		tf.summary.scalar('losses/Total_Loss', total_loss)
		tf.summary.scalar('accuracy', accuracy)
		tf.summary.scalar('learning_rate', lr)
		my_summary_op = tf.summary.merge_all()

		#Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
		def train_step(sess, train_op, global_step,feed_dict):
		    '''
		    Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
		    '''
		    #Check the time for each sess run
		    start_time = time.time()
		    total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op],feed_dict=feed_dict)
		    time_elapsed = time.time() - start_time

		    #Run the logging to print some results
		    logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

		    return total_loss, global_step_count

		#Now we create a saver function that actually restores the variables from a checkpoint file in a sess
		saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V1)
		def restore_fn(sess):
		    return saver.restore(sess, checkpoint_file)

		#Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
		sv = tf.train.Supervisor(logdir = './log', summary_op = None, init_fn = restore_fn)


		#Run the managed session
		with sv.managed_session() as sess:
		    for step in xrange(num_steps_per_epoch * num_epochs):
		        #At the start of every epoch, show the vital information:
		        #sess.run(train_init_op)
				feed_dict = fill_feed_dict(data_sets.train,images,labels)
				if step % num_batches_per_epoch == 0:
				    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
				    learning_rate_value, accuracy_value = sess.run([lr, accuracy],feed_dict = feed_dict)
				    logging.info('Current Learning Rate: %s', learning_rate_value)
				    logging.info('Current Streaming Accuracy: %s', accuracy_value)

				    # optionally, print your logits and predictions for a sanity check that things are going fine.
				    logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, labels],feed_dict = feed_dict)
				    print 'logits: \n', logits_value
				    print 'Probabilities: \n', probabilities_value
				    print 'predictions: \n', predictions_value
				    print 'Labels:\n:', labels_value

				#Log the summaries every 10 step.
				if step % 10 == 0:
				    loss, _ = train_step(sess, train_op, sv.global_step,feed_dict=feed_dict)
				    summaries = sess.run(my_summary_op,feed_dict=feed_dict)
				    sv.summary_computed(sess, summaries)
				    
				#If not, simply run the training step
				else:
				    loss, _ = train_step(sess, train_op, sv.global_step,feed_dict=feed_dict)

		    #We log the final training loss and accuracy
		    logging.info('Final Loss: %s', loss)
		    logging.info('Final Accuracy: %s', sess.run(accuracy,feed_dict=feed_dict))

		    #Once all the training has been done, save the log files and checkpoint model
		    logging.info('Finished training! Saving model to disk now.')
		    sv.saver.save(sess, "./res101_model.ckpt")
		    sv.saver.save(sess, sv.save_path, global_step = sv.global_step)

		'''
		# Restore only the layers up to fc7 (included)
		# Calling function `init_fn(sess)` will load all the pretrained weights.
		#variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
		init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

		# Initialization operation from scratch for the new "fc8" layers
		# `get_variables` will only return the variables whose name starts with the given pattern
		#fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
		#fc8_init = tf.variables_initializer(fc8_variables)

		# ---------------------------------------------------------------------
		# Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
		# We can then call the total loss easily
		#tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
		#loss = tf.losses.get_total_loss()



		# First we want to train only the reinitialized last layer fc8 for a few epochs.
		# We run minimize the loss only with respect to the fc8 variables (weight and bias).
		#fc8_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate1)
		#fc8_train_op = fc8_optimizer.minimize(loss, var_list=fc8_variables)

		# Then we want to finetune the entire model for a few epochs.
		# We run minimize the loss only with respect to all the variables.
		#full_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate2)
		#full_train_op = full_optimizer.minimize(loss)

		# Evaluation metrics
		#prediction = tf.to_int32(tf.argmax(logits, 1))
		#correct_prediction = tf.equal(prediction, labels)
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




		tf.get_default_graph().finalize()

		# --------------------------------------------------------------------------
		# Now that we have built the graph and finalized it, we define the session.
		# The session is the interface to *run* the computational graph.
		# We can call our training operations with `sess.run(train_op)` for instance
		with tf.Session(graph=graph) as sess:
		init_fn(sess)  # load the pretrained weights
		#sess.run(fc8_init)  # initialize the new fc8 layer

		# Update only the last layer for a few epochs.
		for epoch in range(args.num_epochs1):
		    # Run an epoch over the training data.
		    print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
		    # Here we initialize the iterator with the training set.
		    # This means that we can go through an entire epoch until the iterator becomes empty.
		    sess.run(train_init_op)
		    logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, labels])
		    print 'logits: \n', logits_value
		    print 'Probabilities: \n', probabilities_value
		    print 'predictions: \n', predictions_value
		    print 'Labels:\n:', labels_value
		    #while True:
		    #    try:
		    #        _ = sess.run(fc8_train_op, {is_training: True})
		    #    except tf.errors.OutOfRangeError:
		    #        break

		    # Check accuracy on the train and val sets every epoch.
		    train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
		    val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
		    print('Train accuracy: %f' % train_acc)
		    print('Val accuracy: %f\n' % val_acc)


		# Train the entire model for a few more epochs, continuing with the *same* weights.
		for epoch in range(args.num_epochs2):
		    print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
		    sess.run(train_init_op)
		    while True:
		        try:
		            _ = sess.run(full_train_op, {is_training: True})
		        except tf.errors.OutOfRangeError:
		            break

		    # Check accuracy on the train and val sets every epoch
		    train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
		    val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
		    print('Train accuracy: %f' % train_acc)
		    print('Val accuracy: %f\n' % val_acc)
	    '''

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)