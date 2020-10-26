import unittest
import example_tasks


class TestSpamTask(unittest.TestCase):

    def test_default_prefix(self):

        spam = example_tasks.SpamTask()

        self.assertEqual(spam.prefix, 'st_')

    def test_default_params(self):

        spam = example_tasks.SpamTask()

        self.assertEqual(spam.params['eggs'], [])

    def test_change_prefix(self):

        spam = example_tasks.SpamTask()
        spam.prefix = 'sp_'

        self.assertEqual(spam.prefix, 'sp_')

    def test_change_params(self):

        spam = example_tasks.SpamTask({'st_eggs': 1}) # note the prefix st_

        self.assertEqual(spam.params['eggs'], 1)


if __name__ == '__main__':
    unittest.main()
