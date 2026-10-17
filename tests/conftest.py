def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: long-running replay of saved MATLAB fits; skip with -m 'not slow'",
    )
