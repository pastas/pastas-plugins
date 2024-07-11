Developers
==========

This section is for developers who want to contribute to the Pastas plugins repository.

How do I create my own plugin for Pastas?
-----------------------------------------

1. Fork and clone the `pastas-plugins` repository.
2. Create a new folder under `pastas_plugins/`. E.g. `pastas-plugins/my_plugin/`.
3. Your folder contents should look something like this::

    pastas_plugins/
        my_plugin/         # your plugin folder
            __init__.py    # init importing version and main functions
            my_plugin.py   # your awesome contribution to Pastas here
            version.py     # add a version number here

4. Under `docs/plugins/` create a new `.rst` file for your plugin. E.g. `my_plugin.rst`.
   Write up your documentation, and use the autodoc directives to include the 
   documentation of your custom classes/functions. See the other plugin examples 
   for inspiration.
5. Under `docs/examples` add a notebook showcasing your plugin. E.g. `my_plugin.ipynb`.
   Add a link to this notebook in `docs/examples/index.rst`.
6. Under `tests/` add a script, e.g. `test_my_plugin.py` that tests your code.
7. In `pyproject.toml` add a dependency group under `[project.optional-dependencies]`.
   Add any dependencies your plugin relies on. Don't forget to add version numbers if 
   necessary.
8. Commit and push your changes to your fork and submit a Pull Request to the main 
   repository. Your contribution will be reviewed by the Pastas development team. Once
   it has been accepted, your plugin will be available to the Pastas community.
9. Celebrate! You are now a Pastas plugin developer!


Requirements
------------
Your plugin should adhere to the following requirements:

- Your plugin should be compatible with the latest version of Pastas.
- Your plugin should be well-documented.
- Your plugin should be tested. No tests means we will not review your contribution!
- Include an example notebook showcasing your plugin.
