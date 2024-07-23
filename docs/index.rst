Pastas Plugins
==============

Welcome to the pastas-plugins repository, your one-stop-shop for customized,
cutting-edge additions to Pastas.

Pastas plugins are a way to extend the functionality of Pastas. This repository aims to 
provide a template for developing, documenting and testing plugins so that they can be 
shared with the Pastas community.

Check out the documentation for the plugins that are currently included below, or check
out the :ref:`Examples` section for notebooks showcasing the plugins in action.

Current plugins
---------------
The following plugins are currently available:

- :ref:`Cross-correlation <CrossCorrelation>`: analyze and visualize the cross-correlation between two time series.
- :ref:`Modflow <modflow>`: using a Modflow model as a response function in Pastas.
- :ref:`Reservoirs <reservoirs>`: use reservoir models to simulate time series.
- :ref:`Responses <responses>`: extra response functions for Pastas.

Installation
------------
To install the plugins, you can use pip::
   
      pip install pastas-plugins

If you want to use a specific plugin and want to ensure you install all the requisite
dependencies, you can use the following command::

      pip install pastas-plugins[<name of plugin>]

If you want to install them all::
      
            pip install pastas-plugins[all]

Usage
-----

Import the pastas-plugins module with:

.. code-block:: python

      import pastas_plugins as pp

This gives you access to the following functions:

.. code-block:: python

      pp.list_plugins()         # list of all plugins
      pp.show_plugin_versions() # show plugin versions 

The function `pp.show_plugin_versions()` will indicate if any dependencies are missing
for a particular plugin. See the :ref:`index:Installation` section above for tips on how to
install dependencies for a particular plugin.

To use a particular plugin, you'll have to import it explicitly, e.g.:

.. code-block:: python

      from pastas_plugins import responses

      rfunc = responses.Theis()

Separate plugins are each stored in a separate submodule within the pastas-plugins
package. You do not need to install the dependencies for each plugin if you're only
interested in one particular plugin.


Write your own plugin
---------------------
Interested in writing your own plugin? See the :ref:`developers:Developers` section for
a guide on how to get started.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Plugins <plugins/index>
   Examples <examples/index>
   Developers <developers>