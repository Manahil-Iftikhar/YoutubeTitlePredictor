"""Offline, descriptive engagement experiment. No network or import-time training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ['title_length', 'category']


def prepare(frame):
    required = ['title', 'category', 'publishedAt', 'views', 'likes', 'comments']
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    data = frame.copy()
    data['title'] = data['title'].fillna('').str.strip()
    data['category'] = data['category'].fillna('Unknown').astype(str)
    data['publishedAt'] = pd.to_datetime(data['publishedAt'], utc=True, errors='coerce')
    for col in ['views', 'likes', 'comments']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    data = data[(data['title'] != '') & (data[['views', 'likes', 'comments']] >= 0).all(axis=1)]
    # No video IDs are supplied; this is an explicitly imperfect duplicate proxy.
    data = data.drop_duplicates(subset=['title', 'publishedAt']).copy()
    data['title_length'] = data['title'].str.len()
    data['engagement'] = data[['views', 'likes', 'comments']].sum(axis=1)
    return data.sort_values(['publishedAt', 'title']).reset_index(drop=True)


def split_by_date(data):
    dates = sorted(data['publishedAt'].dt.date.unique())
    if len(dates) < 3:
        raise ValueError('At least three distinct publication dates required')
    cut = dates[max(1, int(len(dates) * 0.8))]
    train = data[data['publishedAt'].dt.date < cut]
    test = data[data['publishedAt'].dt.date >= cut]
    if len(train) < 5 or len(test) < 2:
        raise ValueError('Need at least five training and two test rows')
    return train, test


def run(csv_path, output):
    csv_path, output = Path(csv_path), Path(output)
    raw = pd.read_csv(csv_path)
    clean = prepare(raw)
    train, test = split_by_date(clean)
    preprocessing = ColumnTransformer([
        ('length', StandardScaler(), ['title_length']),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['category']),
    ])
    # Hyperparameter is fixed in advance; no tuning on the held-out set.
    models = {
        'median_baseline': DummyRegressor(strategy='median'),
        'ridge_log_target': TransformedTargetRegressor(
            regressor=make_pipeline(preprocessing, Ridge(alpha=10.0)),
            func=np.log1p, inverse_func=np.expm1),
    }
    results = {}
    predictions = pd.DataFrame({'publishedAt': test['publishedAt'], 'actual_engagement': test['engagement']})
    for name, model in models.items():
        model.fit(train[FEATURES], train['engagement'])
        pred = np.maximum(0, model.predict(test[FEATURES]))
        predictions[name] = pred
        results[name] = {'mae': float(mean_absolute_error(test['engagement'], pred)),
                         'rmse': float(np.sqrt(mean_squared_error(test['engagement'], pred))),
                         'r2': float(r2_score(test['engagement'], pred))}
    report = {
        'source_file': csv_path.name, 'source_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'raw_rows': len(raw), 'clean_rows': len(clean), 'train_rows': len(train), 'test_rows': len(test),
        'train_latest_publication': str(train['publishedAt'].max()),
        'test_earliest_publication': str(test['publishedAt'].min()),
        'features': FEATURES, 'target': 'snapshot views + likes + comments',
        'split': 'earliest 80% of distinct publication dates for training; remaining dates held out',
        'versions': {'pandas': pd.__version__, 'numpy': np.__version__, 'scikit_learn': sklearn.__version__},
        'metrics': results,
        'limitations': ['Only a small trending-video snapshot; collection time and video IDs absent',
                        'Publication order is not observation time; this is not future engagement forecasting',
                        'Popularity is confounded by exposure age, channel, category and selection bias',
                        'No title generation, SEO uplift, causal effect, or deployment-quality claim'],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    predictions.to_csv(output / 'predictions.csv', index=False)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(Path(__file__).resolve().parents[1] / 'youtube_video_metadata_cleaned.csv'))
    parser.add_argument('--output', default=str(Path(__file__).resolve().parents[1] / 'reports/engagement'))
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output), indent=2))
