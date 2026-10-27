import gzip
import json
from pathlib import Path
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parent
URLS = {
    'Video_Games_5.json.gz': 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz',
    'meta_Video_Games.json.gz': 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz',
}


def download():
    directory = ROOT / 'data'
    directory.mkdir(exist_ok=True)
    for name, url in URLS.items():
        path = directory / name
        if not path.exists():
            partial = path.with_suffix('.part')
            urllib.request.urlretrieve(url, partial)
            partial.replace(path)


def records(name):
    with gzip.open(ROOT / 'data' / name, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def prepare():
    download()
    fields = ['reviewerID', 'asin', 'overall', 'unixReviewTime']
    raw = pd.DataFrame([{key: row.get(key) for key in fields} for row in records('Video_Games_5.json.gz')])
    raw.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    missing = raw.isna().sum().to_dict()
    valid = raw.user_id.map(lambda x: isinstance(x, str) and bool(x)) & raw.item_id.map(lambda x: isinstance(x, str) and bool(x))
    valid &= raw.rating.isin([1, 2, 3, 4, 5]) & pd.to_numeric(raw.timestamp, errors='coerce').gt(0)
    clean = raw[valid].sort_values(['timestamp', 'user_id', 'item_id'], kind='stable')
    duplicates = int(clean.duplicated(['user_id', 'item_id']).sum())
    clean = clean.drop_duplicates(['user_id', 'item_id'], keep='first').reset_index(drop=True)
    times = clean.timestamp.to_numpy()
    first, second = int(times[int(.8 * len(times))]), int(times[int(.9 * len(times))])
    train_raw = clean[clean.timestamp < first]
    validation_raw = clean[(clean.timestamp >= first) & (clean.timestamp < second)]
    test_raw = clean[clean.timestamp >= second]
    counts = train_raw.groupby('item_id').size().rename('count').reset_index().sort_values(['count', 'item_id'], ascending=[False, True])
    item_ids = sorted(counts.head(5000).item_id)
    candidate = train_raw[train_raw.item_id.isin(item_ids)]
    users = candidate.groupby('user_id').size()
    user_ids = sorted(users[users >= 5].index)
    selected = clean[clean.item_id.isin(item_ids) & clean.user_id.isin(user_ids)].copy()
    user_map, item_map = {key: i for i, key in enumerate(user_ids)}, {key: i for i, key in enumerate(item_ids)}
    selected['user'] = selected.user_id.map(user_map)
    selected['item'] = selected.item_id.map(item_map)
    train = selected[selected.timestamp < first].copy()
    validation = selected[(selected.timestamp >= first) & (selected.timestamp < second)].copy()
    test = selected[selected.timestamp >= second].copy()
    exclusions = []
    for label, frame in [('validation', validation_raw), ('test', test_raw)]:
        new_user = ~frame.user_id.isin(train_raw.user_id)
        new_item = ~frame.item_id.isin(train_raw.item_id)
        retained = frame.user_id.isin(user_ids) & frame.item_id.isin(item_ids)
        exclusions.append({'period': label, 'raw_interactions': len(frame), 'new_user_interactions': int(new_user.sum()),
                           'new_users': frame.loc[new_user, 'user_id'].nunique(), 'new_item_interactions': int(new_item.sum()),
                           'new_items': frame.loc[new_item, 'item_id'].nunique(), 'excluded_total': int((~retained).sum()),
                           'retained_interactions': int(retained.sum())})
    metadata = pd.DataFrame([{key: row.get(key) for key in ['asin', 'title', 'category', 'description', 'brand']}
                             for row in records('meta_Video_Games.json.gz')])
    metadata_duplicates = int(metadata.duplicated('asin').sum())
    metadata = metadata.drop_duplicates('asin', keep='first').set_index('asin').reindex(item_ids)
    coverage = {column: float(metadata[column].map(lambda x: isinstance(x, (str, list)) and len(x) > 0).mean()) for column in metadata}
    info = {'raw_rows': len(raw), 'missing': missing, 'invalid_rows': int((~valid).sum()), 'duplicate_pairs': duplicates,
            'clean_rows': len(clean), 'selected_rows': len(selected), 'users': len(user_ids), 'items': len(item_ids),
            'train_rows': len(train), 'validation_rows': len(validation), 'test_rows': len(test),
            'first_cutoff': first, 'second_cutoff': second, 'sparsity': 1 - len(selected) / (len(user_ids) * len(item_ids)),
            'metadata_duplicate_ids': metadata_duplicates, 'metadata_coverage': coverage, 'exclusions': exclusions}
    return raw, selected, train, validation, test, metadata, info


if __name__ == '__main__':
    *_, info = prepare()
    print(json.dumps(info, indent=2, ensure_ascii=False))
