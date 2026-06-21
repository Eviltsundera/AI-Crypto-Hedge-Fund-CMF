from ai_crypto_hedge_fund import __version__, project_root


def test_package_importable() -> None:
    assert __version__
    assert project_root().name == "AI-Crypto-Hedge-Fund-CMF"
