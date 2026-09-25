import unittest

import numpy as np

from animalclef import add_local_scores, clusters, graph_clusters, hierarchy, predict


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
        graph_groups = graph_clusters((query @ query.T + 1) / 2, 0.9, 2)
        self.assertEqual(graph_groups[0], graph_groups[1])
        self.assertEqual(graph_groups[2], graph_groups[3])
        self.assertNotEqual(graph_groups[0], graph_groups[2])

    def test_local_matches_raise_only_the_matched_pair(self):
        ids = np.array(["a", "b", "c"])
        similarity = np.full((3, 3), 0.7)
        revised = add_local_scores(similarity, ids, ids,
                                   [("a", "b", 80), ("b", "c", 12)], 0.12)
        self.assertAlmostEqual(revised[0, 1], 0.82)
        self.assertAlmostEqual(revised[1, 0], 0.82)
        self.assertAlmostEqual(revised[1, 2], 0.7)


if __name__ == "__main__":
    unittest.main()
