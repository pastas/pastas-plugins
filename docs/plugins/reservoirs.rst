.. _Reservoirs:

Reservoirs
----------

The reservoirs plugin contains reservoir models to model time series. These models
act like Pastas StressModels, but do not use response functions or convolution.

Currently the following reservoir models are available:

- :class:`~pastas_plugins.reservoirs.reservoir.Reservoir1`: A single reservoir, equivalent to the Exponential response function in Pastas.
- :class:`~pastas_plugins.reservoirs.reservoir.Reservoir2`: A single reservoir with two outflow levels.

These can be used in Pastas models with this StressModel:

- :class:`~pastas_plugins.reservoirs.stressmodels.ReservoirModel`: A stressmodel that uses a reservoir model to simulate time series.

Example
^^^^^^^

See the :ref:`Examples` section for more information on how to use the reservoirs plugin.

API
^^^
.. automodule:: pastas_plugins.reservoirs.reservoir
   :members:
   :undoc-members:
   :private-members:

.. automodule:: pastas_plugins.reservoirs.stressmodels
   :members:
   :undoc-members:
   :private-members: