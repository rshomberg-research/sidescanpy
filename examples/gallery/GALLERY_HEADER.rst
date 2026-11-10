Examples
========

Worked walkthroughs on real data. The Guide explains how each part of
sidescanpy works; these pages show it doing a job, and :doc:`/theory/index`
explains why the steps are there.

Each example below runs sidescanpy on a real, openly licensed recording
that anyone can download, and shows the code, what it prints and the
figure it makes. The pages are built by running the scripts in
``examples/gallery/``. Two more sections sit alongside them:

- :doc:`/atr`: two-stage target recognition step by step, from raw data
  or a target catalog to a trained detector and classifier. Those
  scripts need a GPU and extra packages, so they are not run when this
  site is built; each page shows the results of a run.
- :doc:`/example_data`: every recording the examples use, with its
  source, size and licence.

.. toctree::
   :hidden:

   Target recognition </atr>
   Example data </example_data>

Run one yourself with an output directory (the current one by default)::

    python examples/gallery/plot_correct_a_line.py out/

Plotting needs the ``viz`` extra and the GeoTIFF in
:ref:`sphx_glr_examples_plot_grid_to_a_map.py` the ``geo`` extra. The
recordings, their sizes and licences are listed in :doc:`/example_data`.
