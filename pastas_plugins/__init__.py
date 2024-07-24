# ruff : noqa: F401
import pathlib
from importlib import import_module

# from platform import python_version
# from packaging.version import parse as parse_version
from pastas_plugins.version import __version__


def list_plugins():
    plugins = pathlib.Path(__file__).parent.iterdir()
    plugins = [
        plugin.stem
        for plugin in plugins
        if plugin.is_dir() and not plugin.stem.startswith("_")
    ]
    plugins.sort()
    return plugins


def show_plugin_versions():
    showtip = False
    plugins = list_plugins()
    msg = f"pastas_plugins version      : {__version__}\n"
    for plugin in plugins:
        try:
            module = import_module(f"pastas_plugins.{plugin}.version")
            version = module.__version__
        except ModuleNotFoundError:
            showtip = True
            version = "not available (check dependencies)"
        msg += f"- {(plugin + ' version'):25s} : {version}\n"
    if showtip:
        msg += "\nNote: To install missing dependencies use `pip install pastas-plugins[<plugin-name>]`"
    print(msg)
