# Contribution Guidelines

Contributions are welcome! If you have a new conversion option, feature, or other you would like to add, 
so that the whole community can benefit, please open a pull request! We are always looking for new 
conversion options, and we are happy to help you get started with adding a new conversion option/feature!

### Developing alma

Raise an issue before starting to work on a PR. There is also a backlog 
(https://github.com/saifhaq/alma/issues) of issues which are tagged with the area of focus, a 
coarse priority level and whether the issue may be accessible to new contributors. Reach out if 
you are interested in working on a issue, and we can help you get started.

#### Development environment

We recommend doing all development inside of the provided Docker container, i.e. the latest
`saifhaq/alma:latest` image. This ensures that the development environment is consistent across
all contributors. The Docker container is built with all of the necessary dependencies to support
each conversion method.

If you are adding a new dependency to the project (e.g. to enable a new conversion method), please 
update the Dockerfile to include the new dependency as part of your PR.

#### Communication

The primary location for discussion is GitHub issues and Github discussions. This is the best 
place for questions about the project and discussion about specific issues.

Feel free to join the [`alma` Discord channel](https://discord.gg/RASFKzqgfZ), it is a great place 
to ask questions, get help, and dicuss the project and all things conversion and NN acceleration.

### Coding Guidelines


- Formatting your code is essential to ensure code matches the style guidelines. We follow black and isort.
Black: Ensures consistency following a strict subset of PEP 8.
isort: Organizes Python imports systematically.

Once the above dependencies are installed, from the root of the repository, run:

```bash
pre-commit install
```

Once that has been done (and all dependencies installed), `git commit` command will perform 
formatting/linting before committing your code.

- Avoid introducing unnecessary complexity into existing code so that maintainability and 
readability are preserved. In the case of new conversion methods, try to keep each method as independent
as possible. That way others can easily understand the code. It's not just about 
functionality, but about helping people understand how to implement any given conversion method.

- Try to avoid committing commented out code.

- If you submit a new conversion method, please add a test for it in the `tests` directory. It should
also successfuly pass the unit tests. If it doesn't pass the tests, but there is another good 
reason to include it, please explain in the PR.

- Comment subtleties and design decisions.

- Add docstrings to all functions and classes, and type hinting.

- Document hacks, we can discuss it only if we can find it.

### Commits and PRs

- Try to keep pull requests focused (multiple pull requests are okay). Typically PRs should focus 
on a single issue or a small collection of closely related issue.

- Typically we try to follow the guidelines set by https://www.conventionalcommits.org/en/v1.0.0/ 
for commit messages for clarity and semantic versioning. Again not strictly enforced. Generally,
if you are adding a new converison method, the commit message should be prefixed with `feat:` and the
PR should be labelled as minor. Breaking changes should be labelled as major.

### Naming Conventions
If adding new conversion options, please follow the naming conventions outlined in the README [Naming 
Conventions](#naming-conventions) section.


## Structure of the repo

| Component                | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| [**src/alma**](src/alma) | Main codebase, where all conversion options and benchmark utils are implemented |
| [**examples**](examples) | Example applications to show different features of alma |
| [**scripts**](scripts)   | Shell scripts for building and running env and runner Docker images |
| [**docker**](docker)     | Shell script, entry point for CI runner. |
| [**tests**](tests)       | Unit tests for alma                                |

Thanks in advance for your contribution to alma! 🎉
