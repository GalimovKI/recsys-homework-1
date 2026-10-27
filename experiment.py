import itertools
import json
import time
from pathlib import Path
import pandas as pd
from threadpoolctl import threadpool_limits
from data_pipeline import prepare
from evaluation import Evaluation
from features import examples
from models import ItemCF, EASE, IALS, CustomALS, FM

ROOT = Path(__file__).resolve().parent
CLASSES = {'ItemCF': ItemCF, 'EASE': EASE, 'iALS': IALS, 'CustomALS': CustomALS, 'FM': FM, 'FM_ID': FM}


def grid():
    values = [('ItemCF', {'neighbors': k}) for k in [50,100,200]]
    values += [('EASE', {'reg': reg}) for reg in [10,100,1000]]
    for name in ['iALS','CustomALS']:
        values += [(name, dict(factors=f,reg=r,alpha=a,iterations=15))
                   for f,r,a in itertools.product([32,64],[.1,1.],[10,40])]
    for name in ['FM','FM_ID']:
        values += [(name, dict(factors=f,reg=r,metadata=name=='FM',epochs=20))
                   for f,r in itertools.product([16,32],[1e-5,1e-4])]
    return values


def run():
    destination = ROOT / 'results'
    destination.mkdir(exist_ok=True)
    _, _, train, validation, test, _, info = prepare()
    shape = (info['users'],info['items'])
    validation_evaluator = Evaluation(train,validation,shape)
    rows = []
    prepared = None
    with threadpool_limits(limits=1):
        for name, params in grid():
            print(f'SEARCH {name} {params}',flush=True)
            start = time.perf_counter()
            model = CLASSES[name](**params)
            if name.startswith('FM'):
                if prepared is None:
                    prepared = examples(train,shape)
                model.fit(train,shape,prepared=prepared,validation=lambda m: validation_evaluator.evaluate(m)['NDCG@10'])
            else:
                model.fit(train,shape)
            seconds = time.perf_counter()-start
            result = validation_evaluator.evaluate(model)
            row = {'model':name,'params':params,'fit_seconds':seconds,**result}
            if name.startswith('FM'):
                row.update(best_epoch=model.best_epoch,history=model.history)
            rows.append(row)
            print({k:v for k,v in row.items() if k!='history'},flush=True)
            del model
        selected = {}
        for name in CLASSES:
            candidates = [row for row in rows if row['model']==name]
            selected[name] = max(candidates,key=lambda r:r['NDCG@10'])
        # Choose all settings before evaluating on test.
        specifications = {name: dict(row['params']) for name,row in selected.items()}
        for name in ['FM','FM_ID']:
            specifications[name]['epochs'] = selected[name]['best_epoch']
        specifications['CustomALS_matched'] = specifications['iALS'].copy()
        specifications['FM_ID_matched'] = {**specifications['FM'],'metadata':False}
        combined = pd.concat([train,validation],ignore_index=True)
        evaluator = Evaluation(combined,test,shape)
        prepared_final = None
        final = []
        recommendations = {}
        computed = {}
        for name, params in specifications.items():
            class_name = name.replace('_matched','')
            key = (class_name,json.dumps(params,sort_keys=True))
            if key in computed:
                final.append({**computed[key],'model':name})
                continue
            print(f'FINAL {name} {params}',flush=True)
            start = time.perf_counter()
            model = CLASSES[class_name](**params)
            if class_name.startswith('FM'):
                if prepared_final is None:
                    prepared_final = examples(combined,shape)
                model.fit(combined,shape,prepared=prepared_final)
            else:
                model.fit(combined,shape)
            seconds = time.perf_counter()-start
            result = evaluator.evaluate(model)
            row = {'model':name,'params':params,'fit_seconds':seconds,**result}
            final.append(row)
            computed[key] = row
            recommendations[name] = model.recommend(evaluator.users[:3],evaluator.seen,10).tolist()
            print(row,flush=True)
            del model
    result = {'seed':42,'data':info,'validation_users':len(validation_evaluator.users),
              'test_users':len(evaluator.users),'validation':rows,'selected':selected,'test':final,
              'example_users':evaluator.users[:3].tolist(),'recommendations':recommendations}
    (destination / 'experiment.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    pd.DataFrame([{**row,'params':json.dumps(row['params'],sort_keys=True)} for row in final]).to_csv(destination/'test_metrics.csv',index=False)
    pd.DataFrame([{k:(json.dumps(v,sort_keys=True) if k=='params' else v) for k,v in row.items() if k!='history'} for row in rows]).to_csv(destination/'validation_metrics.csv',index=False)
    return result


if __name__ == '__main__':
    run()
