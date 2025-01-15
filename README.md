# pastas-plugins

Welcome to the pastas-plugins repository, your one-stop-shop for customized
cutting-edge additions to Pastas.


## Current plugins

The following plugins are now available:

- **`cross_correlation`**: analyze and visualize the cross-correlation between two time series.
- **`modflow`**: use modflow models as response functions.
- **`reservoirs`**: use reservoir models to simulate time series.
- **`responses`**: custom response functions for Pastas.
- **`pest`**: PEST(++) solver for Pastas.

## Installation

<!-- TODO: add repo to PYPI so this becomes true: -->
Install `pastas-plugins` with:

```bash
pip install pastas_plugins
```

If you want to use a specific plugin and want to ensure you install all the requisite
dependencies, you can use the following command:

```bash
pip install pastas-plugins[<name of plugin>]
```

If you want to install them all:
```bash
pip install pastas-plugins[all]
```

## Usage

Import the pastas-plugins module with:

```python
import pastas_plugins as pp
```

This gives you access to the following functions:

```python
pp.list_plugins()         # list of all plugins
pp.show_plugin_versions() # show plugin versions
```

The function `pp.show_plugin_versions()` will indicate if any dependencies are missing
for a particular plugin. See the [Installation](#installation) section above for tips
on how to install dependencies for a particular plugin.

To use a particular plugin, you'll have to import it explicitly, e.g.:

```python
from pastas_plugins import responses

rfunc = responses.Theis()
```

Separate plugins are each stored in a separate submodule within the pastas-plugins
package. You do not need to install the dependencies for each plugin if you're only
interested in one particular plugin.