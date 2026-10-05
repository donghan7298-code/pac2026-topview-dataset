"""Create separate GitHub source and generated-data archives."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path


def package(dataset,labels,model,out):
    root=Path(__file__).parents[1];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dataset,labels,model=Path(dataset),Path(labels),Path(model)
    lm=json.loads((labels/'labels_manifest.json').read_text())
    summary=dict(source='SYNTHETIC_DEVELOPMENT_ONLY',splits=lm['splits'],
        total_states=sum(s.get('states',0) for s in lm['splits'].values()),
        total_candidate_targets=sum(s.get('candidate_targets',0) for s in lm['splits'].values()),
        feature_count=len(lm['feature_names']),label_kind=lm['label_kind'])
    (root/'docs'/'GENERATED_LABELS_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(model/'training_report.json',root/'docs'/'EXAMPLE_TRAINING_REPORT.json')
    shutil.copy2(model/'ranker_model.json',root/'examples'/'ranker_model.json')
    shutil.copy2(dataset/'validation_report.json',root/'docs'/'VALIDATION_REPORT.json')
    if (model/'paired_comparison.json').exists():
        shutil.copy2(model/'paired_comparison.json',root/'docs'/'EXAMPLE_COMPARISON_REPORT.json')
    source_files=[p for p in root.rglob('*') if p.is_file() and not any(part in ('__pycache__','data','.git') for part in p.relative_to(root).parts) and p.suffix not in ('.pyc','.zip')]
    source_zip=out/'PAC2026_topview_github_ready.zip'
    with zipfile.ZipFile(source_zip,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(source_files):z.write(p,str(Path(root.name)/p.relative_to(root)))
    data_zip=out/'PAC2026_topview_training_data.zip'
    with zipfile.ZipFile(data_zip,'w',zipfile.ZIP_DEFLATED) as z:
        for folder,name in [(dataset,'generated'),(labels,'labels'),(model,'model')]:
            for p in sorted(folder.iterdir()):
                if p.is_file():z.write(p,str(Path('data')/name/p.name),compress_type=zipfile.ZIP_STORED if p.suffix=='.gz' else zipfile.ZIP_DEFLATED)
    for p in [source_zip,data_zip]:
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
        print(f'{p.name}: {p.stat().st_size:,} bytes')
    print(json.dumps(summary,ensure_ascii=False))
    return source_zip,data_zip


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='data/generated');p.add_argument('--labels',default='data/labels');p.add_argument('--model',default='data/model');p.add_argument('--out',default='../deliverables');a=p.parse_args();package(a.dataset,a.labels,a.model,a.out)
