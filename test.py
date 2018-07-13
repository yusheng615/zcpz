import unittest
import xmlrunner
from code import *
from _math import *
from random_mpt import resampleportfolio
import numpy as np


class OperationFuncsTest(unittest.TestCase):

    def setUp(self):
        n_assets = 5
        n_obs = 252
        np.random.seed(1)
        self.returns = np.random.randn(n_obs, n_assets)

    def tearDown(self):
        print('代码测试结束')

    def test_add(self):
        self.assertEqual(add(4, 2), 6)
        self.assertTrue(add(1, 1), 1)

    def test_case(self):
        self.assertTrue(is_barcode('126112611262'), 'it is not Barcode')

    def test_init(self):
        d = Dictcase(a=1, b='test')
        self.assertTrue(isinstance(d, dict))

    def test_num(self):
        self.assertEqual(three_num(), 36)

    def test_sqrt(self):
        self.assertEqual(find_sqrt(), (11, 17))

    def test_custom_range(self):
        self.assertTrue(isinstance(custom_range(0, -5), list))

    def test_random(self):
        self.assertTrue(resampleportfolio(self.returns), 10)



if __name__ == '__main__':
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='./mk_xml'))

