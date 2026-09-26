import tempfile
import unittest
from pathlib import Path
import pandas as pd
from experiments.engagement_baseline import prepare, split_by_date, run

class ExperimentTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([dict(title=f'Title {i}', category='Example', publishedAt=f'2024-01-{i+1:02d}', views=i+10, likes=2, comments=1) for i in range(12)])

    def test_clean_invalid_and_duplicates(self):
        data = self.fixture()
        data.loc[0, 'views'] = -1
        data.loc[1, 'publishedAt'] = 'bad'
        data = pd.concat([data, data.iloc[[4]]], ignore_index=True)
        self.assertEqual(len(prepare(data)), 10)

    def test_dates_never_overlap(self):
        data = prepare(self.fixture())
        train, test = split_by_date(data)
        self.assertLess(train.publishedAt.max(), test.publishedAt.min())
        self.assertFalse(set(train.publishedAt.dt.date) & set(test.publishedAt.dt.date))

    def test_missing_schema_and_small_data_fail(self):
        with self.assertRaises(ValueError): prepare(pd.DataFrame({'title': []}))
        with self.assertRaises(ValueError): split_by_date(prepare(self.fixture().iloc[:2]))

    def test_reproducible_reports_and_no_count_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.csv'; self.fixture().to_csv(path, index=False)
            a = run(path, Path(tmp)/'a'); b = run(path, Path(tmp)/'b')
            self.assertEqual(a, b)
            self.assertEqual(a['features'], ['title_length', 'category'])
            self.assertEqual(set(a['metrics']), {'median_baseline', 'ridge_log_target'})
            self.assertTrue((Path(tmp)/'a/predictions.csv').exists())

if __name__ == '__main__': unittest.main()
