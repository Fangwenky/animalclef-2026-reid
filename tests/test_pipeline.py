import unittest

import numpy as np

from animalclef import clusters, hierarchy, predict


class PipelineTest(unittest.TestCase):
    def test_clusters_and_known_attachment(self):
        query = np.array([[1, 0, 0], [0.99, 0.1, 0], [0, 1, 0], [0, 0.99, 0.1]], dtype=float)
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        groups = clusters(hierarchy(query), len(query), 0.9)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], groups[3])
        self.assertNotEqual(groups[0], groups[2])
        labels = predict(query, np.array([[1, 0, 0]], dtype=float), np.array(["animal-a"]), groups, 0.9)
        self.assertEqual(labels[0], "known:animal-a")
        self.assertEqual(labels[1], "known:animal-a")
        self.assertEqual(labels[2], labels[3])
        self.assertTrue(labels[2].startswith("new:"))


if __name__ == "__main__":
    unittest.main()
