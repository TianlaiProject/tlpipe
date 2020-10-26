import os
import unittest
from tlpipe.pipeline import pipeline


class TestPipeline(unittest.TestCase):

    def test_example1(self):

        pipe = pipeline.Manager('./example1.pipe')
        pipe.run()

    def test_example2(self):

        pipe = pipeline.Manager('./example2.pipe')
        pipe.run()

    def tearDown(self):
        os.system('rm -rf ./output')


if __name__ == '__main__':
    unittest.main()