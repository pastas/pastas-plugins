.. _DataWorth:

Data Worth
----------

.. role:: python(code)
   :language: python

The :code:`dataworth` plugin contains a class and and visualizations to analyze the 
data worth of observations in a pastas model.

Currently the following class is available:

- :class:`~pastas_plugins.dataworth.DataWorth`: Create a DataWorth class to run data worth analysis

This class exposes the following methods to analyze data worth:

- :meth:`~pastas_plugins.dataworth.DataWorth.data_worth_per_observation`: Compute data worth per observation with leave-one-out analysis.
- :meth:`~pastas_plugins.dataworth.DataWorth.data_worth_thinning`: Compute data worth for groups of observations by specifying thinning factors.
- :meth:`~pastas_plugins.dataworth.DataWorth.data_worth_per_added_observation`: Compute data worth for each added observation.
- :meth:`~pastas_plugins.dataworth.DataWorth.data_worth_new_observations`: Compute data worth for groups of added observations.
- :meth:`~pastas_plugins.dataworth.DataWorth.recompute_jacobian`: Recompute jacobian, needed to include new observations for data worth analysis.

These methods are the backbone of the above calculations:

- :meth:`~pastas_plugins.dataworth.DataWorth.observation_noise_covariance`: Compute an observation noise covariance matrix.
- :meth:`~pastas_plugins.dataworth.DataWorth.fisher_information_matrix`: Compute the Fisher information matrix.
- :meth:`~pastas_plugins.dataworth.DataWorth.compute_covariance`: Compute parameter covariance matrix, given a jacobian, an observation noise covariance matrix and an optional mask
- :meth:`~pastas_plugins.dataworth.DataWorth.data_worth`: Compute overall data worth (log determinant of parameter covariance matrix) and per-parameter data worth (variance of parameters).

The following plots are available:

- :func:`~pastas_plugins.dataworth.plot_data_worth_series`: Plot the overall data worth and per parameter data worth for each per observation.
- :func:`~pastas_plugins.dataworth.plot_data_worth_heatmap`: Plot a heatmap of data worth with years on the y-axis and days on the x-axis.

Example
^^^^^^^

See the :ref:`Examples` section for more information on how to use the :code:`dataworth` plugin.

API
^^^
.. automodule:: pastas_plugins.dataworth
   :members:
   :undoc-members:
   :private-members:
