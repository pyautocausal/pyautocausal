# Vendored SyntheticControlMethods backend

This directory contains PyAutoCausal's adapted copy of [SyntheticControlMethods](https://github.com/OscarEngelbrektson/SyntheticControlMethods), originally authored by **Oscar Engelbrektson** (`oscar.engelbrektson@gmail.com`). The runtime code remains in `SyntheticControlMethods/`, with its source notices and the complete [Apache-2.0 license](LICENSE) retained.

Install and develop this backend through the root PyAutoCausal package. Its dependencies are declared in the root [pyproject.toml](../../../../pyproject.toml) and resolved in the root Poetry lockfile. This vendored directory is not a separate distribution; copied standalone build files and upstream demo assets have been removed.

PyAutoCausal includes local correctness fixes to predictor scaling, panel ordering, control selection, treatment boundaries and placebo fitting. See the [correction notes](../../../../CHANGELOG.md) and [method assumptions and limitations](../../../../docs/causal-methods.md#hainmueller-synthetic-control) when using this version.

For a maintained end-to-end example, run the [California Proposition 99 script](../../../../examples/scripts/california_prop99.py) using the [example instructions](../../../../examples/README.md). The [upstream tutorial](https://github.com/OscarEngelbrektson/SyntheticControlMethods/blob/master/examples/user_guide.ipynb) is background material for the separate upstream package; its imports and environment do not describe this adapted backend.
