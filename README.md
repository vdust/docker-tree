Docker tree is a simple command line utility that displays docker images as a tree.

- Multiple tags on the same image are displayed together
- An image built from another pulled image is displayed as a child of that image with net size displayed
- Also list the containers running on each image, with a color-coded state

## How to install

Just use the installation feature of your modern python package manager of choice
(support for `pyproject.toml` file required).

_Example with pip install (for `uv` enthousiasts, just add `uv` in front of the command):_
```
pip install git+https://github.com/vdust/docker-tree.git
```

The `docker-tree` command should then be available in the bin folder of your python environment.

### I don't know what a package manager is

In that case, you can simply copy the script `docker_tree.py` and run it directly with python:
```
python docker_tree.py
```

If the `docker` package is installed in your environment, it will be used, otherwise it will attempt to use the
`docker` command line client _(way slower, and net sizes will also be inaccurate due to implementation choices to speed
things up as much as possible)_.

## How to use

```
docker-tree
```

That's it. The script will connect to the docker daemon set in the `DOCKER_HOST` environment variable, falling back to
the local unix socket connexion if not set.

### Sort images

By default, images are sorted by tag names in each branch of the tree.

The `-s` option (or `--size`) sorts images by size of the largest descendant image, in each branch of the tree.

The `-r` option (or `--reverse`) reverse the sort order.

### Highlight images

Optional regular expressions can be provided as arguments. Each image id, image name or container name that match any of
the regular expressions will then be highlighted in the tree

```
docker-tree python
```

### I don't like colors

You can disable output coloring by setting the `USECOLOR` environment variable to `0`
```
USECOLOR=0 docker-tree
```
