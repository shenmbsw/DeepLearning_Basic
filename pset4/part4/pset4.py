import tensorflow as tf
import numpy as np
import read_data
import model


def visualize(model_dict, dataset_generators, epoch_n, print_every,batch_size):
    logs_path = "tmp/SVHN/1"
    tf.reset_default_graph()


    cross_entropy = model_dict['loss']
    accuracy = model_dict['accuracy']

    a = tf.summary.scalar("cost", cross_entropy)
    b = tf.summary.scalar("accuracy", accuracy)
    c = tf.summary.merge([a,b])

    print(a,b,c)
    train_op = model_dict['train_op']

    with model_dict['graph'].as_default():
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            writer =  tf.summary.FileWriter(logs_path,graph = tf.get_default_graph())

            for epoch_i in range(epoch_n):
                for iter_i, data_batch in enumerate(dataset_generators['train']):
                    train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                    _, summary = sess.run([train_op, c], feed_dict=train_feed_dict)
                    writer.add_summary(summary, epoch_i * batch_size + iter_i)

def main():
    # FOR SVHN
    dataset_generators = {
        'train': read_data.svhn_dataset_generator('train', 256),
        'test': read_data.svhn_dataset_generator('test', 256)
    }

    model_dict = model.apply_classification_loss(model.cnn_map)
    visualize(model_dict, dataset_generators, epoch_n=20, print_every=100,batch_size=256)

main()
