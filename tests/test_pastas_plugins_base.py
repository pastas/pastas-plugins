# ruff: noqa: F401


def test_import():
    import pastas_plugins as pp


def test_show_plugin_versions(capsys):
    import pastas_plugins as pp

    pp.show_plugin_versions()

    captured = capsys.readouterr()
    assert "pastas_plugins version" in captured.out