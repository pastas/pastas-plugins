.. _CrossCorrelation:

Cross-correlation
-----------------

.. role:: python(code)
   :language: python

The :code:`cross_correlation` plugin contains functions and visualizations to analyze the 
cross-correlation between two time series.

Currently the following functions are available:

- :func:`~pastas_plugins.cross_correlation.crosscorr.ccf`: Compute the cross-correlation function for two time series.
- :func:`~pastas_plugins.cross_correlation.crosscorr.prewhiten`: Prewhiten time series using an autoregressive model.
- :func:`~pastas_plugins.cross_correlation.crosscorr.fit_response`: Fit Pastas response function to the scaled cross-correlation function.

The following plots are available:

- :func:`~pastas_plugins.cross_correlation.plots.plot_corr`: Plot the cross-correlation result (:python:`ccf(x, y)`) between two time series.
- :func:`~pastas_plugins.cross_correlation.plots.plot_ccf_overview`: Plot an overview of the cross-correlation between two time series.

Example
^^^^^^^

See the :ref:`Examples` section for more information on how to use the reservoirs plugin.

API
^^^
.. automodule:: pastas_plugins.cross_correlation.crosscorr
   :members:
   :undoc-members:
   :private-members:

.. automodule:: pastas_plugins.cross_correlation.plots
   :members:
   :undoc-members:
   :private-members:
