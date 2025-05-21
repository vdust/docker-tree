Docker tree is a simple command line utility that displays docker images as a tree.

- Multiple tags on the same image are displayed together
- An image built from another pulled image is displayed as a child of that image
- Also list the containers running on each image, with a color-coded state

When installed, the script can be called with the command `docker-tree` and will use the package docker for fast query
of images and containers informations.

You can also copy the script and run it directly with python : if `docker` package is installed, it will be used,
otherwise it will use the `docker` command line client as a slower fallback.
