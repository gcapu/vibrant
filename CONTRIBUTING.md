# Contributing

## Setup

To run this project you need poetry.

Poetry: [https://poetry.eustace.io/docs/#installation](https://poetry.eustace.io/docs/#installation)

Install project dependencies into a virtual environment:

```text
poetry install
```

## Development Tasks

Run the tests:

```text
poetry run test
```

Run static analysis:

```text
poetry run lint
```

Run formatting tools analysis:

```text
poetry run format
```

Build the documentation:

```text
poetry run docs
```

To see all available commands, see the `pyproject.toml` file.

## Demo Tasks

Run the program:

```text
poetry run start
```

## Release Tasks

Release to PyPI:

```text
poetry publish
```
