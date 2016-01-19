#!/usr/bin/env python

import unittest
import numpy as np
from toy_datasets import load_digit_dataset
from optimization import mini_batch_sgd, mini_batch_sgd_with_annealing


class skLearnDigitTest(unittest.TestCase):
    def setUp(self):
        tr_data, ts_data = load_digit_dataset(0.7)
        self.tr = np.array([x[0] for x in tr_data])
        self.tr_l = [x[1] for x in tr_data]
        self.xtr = np.array([x[0] for x in ts_data])
        self.xtr_l = [x[1] for x in ts_data]

    def checkModel(self, test_name, model_type, hidden_dim, verbose, epochs, batch_size=10, extra_args=None):
        net, results = mini_batch_sgd(motif=test_name,
                                      train_data=self.tr, labels=self.tr_l,
                                      xTrain_data=self.xtr, xTrain_targets=self.xtr_l,
                                      learning_rate=0.001, L1_reg=0.0, L2_reg=0.0, epochs=epochs,
                                      batch_size=batch_size, hidden_dim=hidden_dim, model_type=model_type,
                                      model_file=None, trained_model_dir=None, verbose=verbose, extra_args=extra_args)
        self.assertTrue(results['batch_costs'][1] > results['batch_costs'][-1])
        self.assertTrue(results['xtrain_accuracies'][1] < results['xtrain_accuracies'][-1])

    def test_twoLayerNeuralNetwork(self):
        self.checkModel(test_name="twoLayerTest", model_type="twoLayer", hidden_dim=[10], verbose=False, epochs=1000)

    def test_threeLayerNeuralNetwork(self):
        self.checkModel(test_name="threeLayerTest", model_type="threeLayer", hidden_dim=[10, 10], verbose=False,
                        epochs=1000)
        self.checkModel(test_name="ReLUthreeLayerTest", model_type="ReLUthreeLayer", hidden_dim=[10, 10],
                        verbose=False, epochs=1000)

    def test_fourLayerNeuralNetwork(self):
        self.checkModel(test_name="fourLayerTest", model_type="fourLayer", hidden_dim=[10, 10, 10],
                        verbose=False, epochs=1000)
        self.checkModel(test_name="ReLUfourLayerTest", model_type="ReLUfourLayer", hidden_dim=[10, 10, 10],
                        verbose=False, epochs=1000)

    def test_ConvNet(self):
        conv_args = {
            "batch_size": 50,
            "n_filters": [5],
            "n_channels": [1],
            "data_shape": [8, 8],    # (8-3+1, 8-3+1) = (6, 6)
            "filter_shape": [3, 3],  # (6/2, 6/2) = (3, 3)
            "poolsize": (2, 2)       # output is (batch_size, n_nkerns[0], 3, 3)
        }
        self.checkModel(test_name="ConvNetTest", model_type="ConvNet3", hidden_dim=10, verbose=True, epochs=1000,
                        extra_args=conv_args, batch_size=conv_args['batch_size'])

    def test_scrambledLabels(self):
        np.random.shuffle(self.xtr)
        net, results = mini_batch_sgd(motif="scrambled",
                                      train_data=self.tr, labels=self.tr_l,
                                      xTrain_data=self.xtr, xTrain_targets=self.xtr_l,
                                      learning_rate=0.001, L1_reg=0.0, L2_reg=0.0, epochs=1000, batch_size=10,
                                      hidden_dim=[10, 10], model_type="threeLayer", model_file=None,
                                      trained_model_dir=None,
                                      verbose=False)
        self.assertTrue(results['batch_costs'][1] > results['batch_costs'][-1])
        self.assertAlmostEqual(results['xtrain_accuracies'][1], results['xtrain_accuracies'][-1], delta=2.5)

    def test_annealingLearningRate(self):
        net, results = mini_batch_sgd_with_annealing(motif="annealing",
                                                     train_data=self.tr, labels=self.tr_l,
                                                     xTrain_data=self.xtr, xTrain_targets=self.xtr_l,
                                                     learning_rate=0.001, L1_reg=0.0, L2_reg=0.0, epochs=100,
                                                     batch_size=10,hidden_dim=[10, 10], model_type="threeLayer",
                                                     model_file=None, trained_model_dir=None, verbose=False)
        self.assertTrue(results['batch_costs'][1] > results['batch_costs'][-1])
        self.assertTrue(results['xtrain_accuracies'][1] < results['xtrain_accuracies'][-1])

# TODO illegal network tests
# TODO dump/load tests
# TODO MNIST DATASET


def main():
    testSuite = unittest.TestSuite()
    testSuite.addTest(skLearnDigitTest('test_twoLayerNeuralNetwork'))
    testSuite.addTest(skLearnDigitTest('test_threeLayerNeuralNetwork'))
    testSuite.addTest(skLearnDigitTest('test_fourLayerNeuralNetwork'))
    testSuite.addTest(skLearnDigitTest('test_ConvNet'))
    testSuite.addTest(skLearnDigitTest('test_scrambledLabels'))
    testSuite.addTest(skLearnDigitTest('test_annealingLearningRate'))

    testRunner = unittest.TextTestRunner(verbosity=2)
    testRunner.run(testSuite)



if __name__ == '__main__':
    main()

