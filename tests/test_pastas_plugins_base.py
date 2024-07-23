# ruff: noqa: F401
from unittest.mock import MagicMock, patch

from pastas_plugins import __version__, show_plugin_versions


def test_import():
    import pastas_plugins as pp

    assert pp is not None


def test_show_plugin_versions_with_available_versions(capsys):
    with patch("pastas_plugins.list_plugins", return_value=["plugin1", "plugin2"]):
        with patch("pastas_plugins.import_module") as mock_import_module:
            version_mock_module1 = "1.0.0"
            version_mock_module2 = "2.0.0"
            mock_module1 = MagicMock()
            mock_module1.__version__ = version_mock_module1
            mock_module2 = MagicMock()
            mock_module2.__version__ = version_mock_module2
            mock_import_module.side_effect = [mock_module1, mock_module2]

            show_plugin_versions()

            captured = capsys.readouterr()
            expected_output1 = f"pastas_plugins version      : {__version__}"
            expected_output2 = f"plugin1 version           : {version_mock_module1}"
            expected_output3 = f"plugin2 version           : {version_mock_module2}"
            assert expected_output1 in captured.out
            assert expected_output2 in captured.out
            assert expected_output3 in captured.out


def test_show_plugin_versions_with_missing_dependencies(capsys):
    with patch("pastas_plugins.list_plugins", return_value=["plugin1"]):
        with patch("pastas_plugins.import_module") as mock_import_module:
            mock_import_module.side_effect = ModuleNotFoundError

            show_plugin_versions()

            captured = capsys.readouterr()
            print(captured.out)
            expected_output1 = f"pastas_plugins version      : {__version__}"
            expected_output2 = (
                "plugin1 version           : not available (check dependencies)"
            )
            expected_output3 = "\nNote: To install missing dependencies use `pip install pastas-plugins[<plugin-name>]`"
            assert expected_output1 in captured.out
            assert expected_output2 in captured.out
            assert expected_output3 in captured.out
