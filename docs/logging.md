# Logging

The library logs through `logging.getLogger("sidescanpy.<module>")` and
never configures handlers itself:

- **DEBUG:** routine progress and decisions (a skipped step, a masked
  region, per-file progress).
- **WARNING:** something about the data the caller should know (missing
  sound velocity, an empty aux stream, a target that could not be
  matched).
- Library code never logs at INFO and never prints.

In a script:

```python
from sidescanpy.log import configure_logging, log_timing

configure_logging("INFO")          # or "DEBUG" to see library detail
with log_timing("correct survey"):
    ...
```

`configure_logging` is idempotent and keeps noisy third-party loggers at
WARNING. The CLI commands set it up from `-v`, `-q` and `--log-file`.
