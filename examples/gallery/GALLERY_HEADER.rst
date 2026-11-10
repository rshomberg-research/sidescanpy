Examples
========

Each example runs sidescanpy on a real, openly licensed recording that
anyone can download, and shows the code, what it prints and the figure
it makes. The pages are built by running the scripts in
``examples/gallery/``.

Run one yourself with an output directory (the current one by default)::

    python examples/gallery/plot_correct_a_line.py out/

Plotting needs the ``viz`` extra and the GeoTIFF in
:ref:`sphx_glr_examples_plot_grid_to_a_map.py` the ``geo`` extra. The
recordings, their sizes and licences are listed in :doc:`/example_data`.

The two-stage target recognition examples (detector, classifier, review,
evaluation) need a GPU and extra packages, so they are not in this
gallery: see :doc:`/atr/index`.
