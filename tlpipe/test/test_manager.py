import unittest
import example_tasks
from tlpipe.pipeline import pipeline


class TestManager(unittest.TestCase):

    def test_default_prefix(self):

        manager = pipeline.Manager({'pipe_copy': False})

        self.assertEqual(manager.prefix, 'pipe_')

    def test_default_params(self):

        manager = pipeline.Manager({'pipe_copy': False})

        self.assertEqual(manager.params['tasks'], [])

    def test_change_prefix(self):

        manager = pipeline.Manager({'pipe_copy': False})
        manager.prefix = 'pp_'

        self.assertEqual(manager.prefix, 'pp_')

    def test_change_params(self):

        manager = pipeline.Manager({'pipe_copy': False, 'pipe_tasks': ['spam']})

        self.assertEqual(manager.params['tasks'], ['spam'])


if __name__ == '__main__':
    unittest.main()
