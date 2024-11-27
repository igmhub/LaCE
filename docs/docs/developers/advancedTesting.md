# MAINTAINING THE AUTOMATED TESTING

LaCE uses automated testing to ensure code quality and prevent regressions. This guide explains how to maintain and extend the test suite. This section is intended for developers who are maintaining the automated testing.

## Running Tests
Automated tests are run using pytest. The tests pipeline is at `.github/workflows/python-tests.yml`. To add another test, you have to:

1. In the section `Run tests`, in `.github/workflows/python-tests.yml`, add the command to run your test.
```yaml
    - name: Run tests
      run: |
        ...
        pytest tests/test_your_test.py
        pytest tests/test_your_other_test.py
```
2. Add the script with your test in the `tests` folder.
3. The testing function must start with `test_` (e.g., `test_my_function`). Tests can take fixtures as arguments.

In the `.github/workflows/python-tests.yml` file, you can specify when the test should be run. For example, currently tests are only run after a PR to the `main` branch.
```yaml
    on:
    push:
        branches: 'main'
```

When a PR is merged into the `main` branch, the tests are run automatically at [Github Actions](https://github.com/igmhub/LaCE/actions).