import tensorflow as tf
import numpy as np
import read_data
import read_cifar10 as cf10
import model


def new_train_model(model_dict, dataset_generators, epoch_n, print_every,
                    save_model=False, load_model=False):
    with model_dict['graph'].as_default(), tf.Session() as sess:
        var_list = [x for x in tf.global_variables()]
        saver = tf.train.Saver([i for i in var_list[:2]])
        if load_model:
            saver.restore(sess,"/usr4/dlearn/shs2016f/cs591s2/pset4/tmp/model.ckpt")
            sess.run(tf.variables_initializer(var_list[2:]))
        else:
            sess.run(tf.global_variables_initializer())

        for epoch_i in range(epoch_n):
            for iter_i, data_batch in enumerate(dataset_generators['train']):
                train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                sess.run(model_dict['train_op'], feed_dict=train_feed_dict)

                if iter_i % print_every == 0:
                    collect_arr = []
                    for test_batch in dataset_generators['test']:
                        test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
                        to_compute = [model_dict['loss'], model_dict['accuracy']]
                        collect_arr.append(sess.run(to_compute, test_feed_dict))
                    averages = np.mean(collect_arr, axis=0)
                    fmt = (epoch_i, iter_i, ) + tuple(averages)
                    print('iteration {:d} {:d}\t loss: {:.3f}, '
                          'accuracy: {:.3f}'.format(*fmt))
        if save_model:
            save_path = saver.save(sess, "/usr4/dlearn/shs2016f/cs591s2/pset4/tmp/model.ckpt")
            print("Model saved in file: %s" % save_path)


def main():
    dataset_generators = {
        'train': read_data.cifar10_dataset_generator('train', 128),
        'test': read_data.cifar10_dataset_generator('test', -1)
    }

    cnn_expanded_dict = model.apply_classification_loss(model.cnn_expanded)

    new_train_model(cnn_expanded_dict, dataset_generators, epoch_n=100,
                print_every=10, save_model=False)

    new_train_model(cnn_expanded_dict, dataset_generators, epoch_n=100,
                print_every=10, load_model=True)

main()
