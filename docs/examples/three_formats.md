# Three formats, one model

Three sonars, three file formats or writers, the same code. Each line is
read with `read_and_convert`, corrected with `correct_survey` (bottom
track, TVG, ground range, range correction) and plotted with
`plot_waterfall`; only the file name, format string and frequency
change.

![Waterfalls from EdgeTech JSF, Benthos XTF and Klein XTF](../assets/gallery/three_formats.png)

From left: EdgeTech (JSF written by Discover) over bedforms and
ridges, Benthos SIS-1624 (XTF written by Isis) over bedforms, and Klein
(XTF written by SonarPro) along a sharp seabed edge in a shallow
estuary, each in dB against across-track and along-track distance.

The Klein file records an altitude of about 50 m in water a few metres
deep, so `bottom_track` is told to replace any recorded altitude more
than 5 m from the tracked one (`max_diff_m`) and to smooth the track
over 21 pings. A few pings still track badly, which shows as white
notches at the top of that panel; the bright horizontal dashes are in
the recorded samples.

```python
--8<-- "examples/gallery/three_formats.py"
```

The last part reads the same EdgeTech pings twice: Discover wrote the
line both as JSF and as XTF. The two readers return identical samples:

```
JSF vs XTF, 288 pings: max |difference| = 0
```

Data: JNCC Wight-Barfleur Reef and mid-Irish Sea (Open Government Licence
v3.0) and USGS Grand Bay (U.S. public domain); see
[the example data](index.md#sources-and-licences).
