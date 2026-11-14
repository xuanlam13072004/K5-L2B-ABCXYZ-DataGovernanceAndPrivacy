"""T03 modules — one file per owner, wired together by src/pipeline.py.

    generate.py     Member 3  synthetic night / glare / fog / blur (+ erase) from dev originals
    provenance.py   Member 2  out/transform_manifest.csv (+ dev gold labels)
    qc.py           Member 4  quality gate on holdout_transformed -> pred_unusable + score
    review_flow.py  Member 5  rank -> contact sheet -> review overrides -> GO/HOLD/REWORK

Keep the function signatures below stable: src/pipeline.py calls them. Change a signature only
together with the Lead (Member 1).
"""
