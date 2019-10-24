# Snorkel Release Guide

## Before You Start

Make sure you have [PyPI](https://pypi.org) and [PyPI Test](https://test.pypi.org/)
accounts with maintainer access to the Snorkel project.
Create a .pypirc in your home directory.
It should look like this:

```
[distutils]
index-servers =
  pypi
  pypitest

[pypi]
username=YOUR_USERNAME
password=YOUR_PASSWORD

[pypitest]
repository=https://test.pypi.org/legacy/
username=YOUR_USERNAME
password=YOUR_PASSWORD
```

Then run `chmod 600 ./pypirc` so only you can read/write.

You'll also need to have permissions to push directly to the `master` branch.


## Release Steps

1. Make sure you're in the top-level `snorkel` directory.
1. Make certain your branch is in sync with head:
   
       $ git pull origin master

1. Make sure `CHANGELOG.md` is up to date for the release: compare against PRs
   merged since the last release & update top heading with release date.

1. Update version to, e.g. 0.9.0 (remove the `+dev` label) in `snorkel/VERSION.py`.

1. Commit these changes and push to master:

       git add . -u
       git commit -m "[RELEASE]: v0.9.0"
       git push origin master

1. Tag the release:

       git tag -a v0.9.0 -m "v0.9.0 release"
       git push origin v0.9.0

1. Build source & wheel distributions:

       rm -rf dist build  # clean old builds & distributions
       python3 setup.py sdist  # create a source distribution
       python3 setup.py bdist_wheel  # create a universal wheel

1. Check that everything looks correct by uploading the package to the PyPI test server:

       twine upload dist/* -r pypitest  # publish to test.pypi.org
       python3 -m venv test_snorkel  # create a virtualenv for testing
       source test_snorkel/bin/activate  # activate virtualenv
       python3 -m pip install -i https://testpypi.python.org/pypi snorkel  # check that install works

1. Publish to PyPI

       twine upload dist/* -r pypi

1. Fork [`conda-forge/snorkel-feedstock`](https://github.com/conda-forge/snorkel-feedstock),
   update the version in
   [`meta.yml`](https://github.com/conda-forge/snorkel-feedstock/blob/master/recipe/meta.yaml),
   submit a PR upstream and follow conda-forge's PR instructions.

1. Copy the release notes in `CHANGELOG.md` to the GitHub tag and publish a release.

1. Update version to, e.g. 0.9.1+dev in `snorkel/VERSION.py`.

1. Add a new changelog entry for the unreleased version:

       ## [Unreleased]
       ## [0.9.1]
       ### [Breaking Changes]
       ### [Added]
       ### [Changed]
       ### [Deprecated]
       ### [Removed]

1. Commit these changes and push to master:

       git add . -u
       git commit -m "[BUMP]: v0.9.1+dev"
       git push origin master
       
1. Add the new tag to [the Snorkel project on ReadTheDocs](https://readthedocs.org/projects/snorkel),
   set it as the default version, and make sure a build is triggered.


## Credit
* [AllenNLP](https://github.com/allenai/allennlp/blob/master/setup.py)
* [Altair](https://github.com/altair-viz/altair/blob/master/RELEASING.md)
