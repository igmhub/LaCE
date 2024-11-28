# MAINTAINING THE DOCUMENTATION

LaCE uses `mkdocs` to build the documentation. The documentation is hosted at [LaCE documentation](https://igmhub.github.io/LaCE/). The documentation can be built locally using the following command:

```bash
mkdocs build
``` 
and then served using

```bash
mkdocs serve
```

The documentation is pushed to the `gh-pages` branch at each release (merge into `main`).
The `gh-pages` branch is automatically updated when a PR is merged into `main`.

In order to write documentation, you can use the following structure:

- `docs/docs/developers`: Documentation for developers
- `docs/docs/users`: Documentation for users

To add a new page, you should create a new `.md` file to the `docs/docs/` folder. 
To define where this document should be included and the structure of the documentation, add the new page to the `mkdocs.yml`. The new page will automatically be added to the navigation menu. 
To have a cleaner structure, add the new page to the corresponding `index.md` file. The documentation is structured with an index file for each section.

