Installation
============

From PyPI
---------

.. code-block:: bash

   pip install quadsv

Two optional extras are available for development and documentation:

.. code-block:: bash

   pip install 'quadsv[dev]'    # tests, linting, jupyter, matplotlib
   pip install 'quadsv[docs]'   # Sphinx + theme + autoapi


From source
-----------

.. code-block:: bash

   git clone https://github.com/JiayuSuPKU/EquivSVT.git
   cd EquivSVT
   pip install -e '.[dev,docs]'


Requirements
------------

- **Python** 3.10+.
- **Runtime dependencies** (installed automatically): ``scanpy``,
  ``spatialdata``, ``finufft``, ``joblib``, ``tqdm``. Through
  ``scanpy`` you also get ``anndata``, ``numpy``, ``scipy``,
  ``scikit-learn`` and ``pandas``.

``spatialdata`` is needed by :class:`~quadsv.DetectorGrid` and
:class:`~quadsv.ComparatorGrid`. ``finufft`` is needed by
:class:`~quadsv.NUFFTKernel`, by :class:`~quadsv.DetectorIrregular`
when you set ``backend="nufft"``, and by
:class:`~quadsv.ComparatorIrregular`.


Verify the install
------------------

.. code-block:: python

   import quadsv

   print(quadsv.__version__)
   print(sorted(quadsv.__all__))

You should see 14 public names organised into four layers (see
:doc:`/guides/quickstart` for what each layer does). The top-level
package is the user-facing surface. The canonical submodule paths
(``quadsv.kernels.*``, ``quadsv.detectors.*``,
``quadsv.comparators.multisample``, ``quadsv.statistics``) are
documented under :doc:`/autoapi/quadsv/index`.