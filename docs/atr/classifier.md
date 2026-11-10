# 3. Train the classifier

The second stage looks closely at each detection and decides what it
is: `MILCO` (mine-like contact), `NOMBO` (non-mine-like bottom object)
or `clutter` (not an object). It sees a 64 px chip, three times the
box's longer side, centred on the box: detail a detector working on
whole images does not use.

The `MILCO` and `NOMBO` chips come from the labelled boxes of the
training years. The `clutter` chips are the detector's own false alarms
on those years ("hard negatives"): what the classifier must reject is
what the detector gets wrong. A first version used random patches of
seabed as clutter and removed only 3 of the 57 false alarms on the test
year.

The model is a ResNet-18 from [torchvision](https://pytorch.org/vision/),
pretrained on ImageNet and fine-tuned for 30 epochs, with the loss
weighting each class by its inverse frequency. With your own survey,
[build_target_dataset](../examples/plot_training_dataset.rst) renders
target chips and background images straight from the sonar data.

## Result

```text
train {'MILCO': 360, 'NOMBO': 229, 'clutter': 1288}
val {'MILCO': 49, 'NOMBO': 0, 'clutter': 55}
best validation accuracy 0.55
```

The validation year has no `NOMBO`, so [step 4](evaluate.md) is the
better measure.

```{literalinclude} ../../examples/atr/atr_3_train_classifier.py
:language: python
```
