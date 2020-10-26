import unittest
from tlpipe.container import raw_timestream


class TestContainer(unittest.TestCase):

    def test_load_data(self):

        rt = raw_timestream.RawTimestream('./example_data.hdf5')
        rt.load_all()

        self.assertIn('vis', rt)
        self.assertEqual(len(rt['vis'].shape), 3)

    def test_rawtimestream_to_timestream(self):

        rt = raw_timestream.RawTimestream('./example_data.hdf5')
        rt.load_all()

        ts = rt.separate_pol_and_bl()

        self.assertIn('vis', ts)
        self.assertEqual(len(ts['vis'].shape), 4)


if __name__ == '__main__':
    unittest.main()
