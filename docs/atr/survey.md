# 6. Run the models on a survey

The trained stages applied to sidescanpy output: swath tiles cut from a
corrected public Klein line (USGS Grand Bay), detections turned into
targets with file, ping, side, sample and position
(`export.ml.boxes_to_targets`), classified, and scored against a catalog
in meters (`targets.match_targets`). The catalog is the five
illustrative picks from
[Target chips](../examples/plot_target_chips.rst).

The models were trained on another sonar (Marine Sonic, 900 kHz)
looking for mine-like objects, so their calls on this line mean little:
this page shows how the pieces connect, not how well these models
transfer. Train on tiles from your own sonar for that.

## Result

```text
2 candidate targets on 4 tiles
min_score 0.05: 0 hit(s), 2 false alarm(s), 5 miss(es) within 3 m: precision 0.00, recall 0.00
min_score 0.25: 0 hit(s), 0 false alarm(s), 5 miss(es) within 3 m: precision nan, recall 0.00
min_score 0.5: 0 hit(s), 0 false alarm(s), 5 miss(es) within 3 m: precision nan, recall 0.00
```

As expected for models trained on another sonar, nothing transfers. The
candidates are still complete targets (`survey_candidates.json`: file,
ping, side, sample, position, score), ready for review and for
`export.ml.build_target_dataset`.

```{literalinclude} ../../examples/atr/atr_6_apply_to_survey.py
:language: python
```
