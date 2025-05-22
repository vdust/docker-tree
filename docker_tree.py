"""Command line utility to view docker images as a tree, with associated containers."""
import dataclasses
import itertools
import json
import logging
import math
import os
import re
import subprocess

from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from typing import Any, ClassVar

try:
    import docker
    docker_interface = 'package'
except ImportError:
    docker_interface = 'cli'

__version__ = "1.0"

ImageDict = dict[str, Any]

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ContainerInfos:
    """Container informations.

    status is one of
        - 'running'
        - 'paused'
        - 'restarting'
        - 'deleting'
        - 'dead'
        - 'created'
        - 'exited-ok'
        - 'exited-ko'
        - 'unknown'
    """

    id: str
    name: str
    image_tag: str
    status: str | None = None
    exit_code: int | None = None
    raw_status: str | None = None

    _exited_re: ClassVar[re.Pattern] = re.compile(r'^exited \((?P<status>-?[0-9]+)\)')
    statuses: ClassVar[tuple[str, ...]] = (
        'created',
        'dead',
        'deleting',
        'exited-ko',
        'exited-ok',
        'paused',
        'restarting',
        'running',
        'unknown',
        )

    def __post_init__(self):
        """Run post initialization."""
        if self.raw_status is not None:
            self.raw_status = self.raw_status.strip().lower()
            if self.raw_status.startswith('up '):
                self.status = 'running' if not self.raw_status.endswith('(paused)') else 'paused'
            elif self.raw_status.startswith('restarting '):
                self.status = 'restarting'
            elif self.raw_status.startswith('removal '):
                self.status = 'deleting'
            elif self.raw_status == 'dead':
                self.status = 'dead'
            elif self.raw_status == 'created':
                self.status = 'created'
            elif self.raw_status.startswith('exited '):
                match = self._exited_re.match(self.raw_status)
                if not match:
                    self.status = 'exited-ko'
                else:
                    self.exit_code = int(match.group('status'))
                    self.status = f"exited-{'ok' if self.exit_code == 0 else 'ko'}"
        if self.status is None:
            self.status = 'unknown'

    def is_running(self):
        """Tell if the instance is running based on status information."""
        return self.status == 'running'


@dataclasses.dataclass
class ImageInfos:
    """Image informations."""
    id: str | None = None
    size: int | None = None
    net_size: int | None = None
    tags: set[str] = dataclasses.field(default_factory=set)
    containers: list[str] = dataclasses.field(default_factory=list)
    layers: list[str] = dataclasses.field(default_factory=list)
    children: dict[str, 'ImageInfos'] | None = None

    _human_bytes_units: ClassVar[list[str]] = ['B', 'kB', 'MB', 'GB', 'TB', 'PB']

    @classmethod
    def get_human_size(cls, size: int | None) -> str:
        """Get size as a human-readable string."""
        scale = 0
        size = size if size is not None else -1
        while size > 1000:
            scale += 1
            size = size / 1000.0
        if size < 0 or scale >= len(cls._human_bytes_units):
            return '???'
        unit = cls._human_bytes_units[scale]
        if size >= 100 or scale == 0:
            return f"{int(size)}{unit}"
        elif size >= 10:
            return f"{size:.01f}{unit}"
        else:
            return f"{size:.02f}{unit}"

    @property
    def first_tag(self) -> str | None:
        """Get first tag in lexicographic order."""
        return None if not self.tags else min(self.tags)

    def sorted_tags(self, reverse: bool = False) -> list[str]:
        """Return a list of tags, sorted by (name, version) tuple."""
        return sorted(self.tags, key=lambda tag: tag.split(':'), reverse=reverse)

    @property
    def sort_size(self) -> str:
        """Get the size of the largest image, children included."""
        size = self.size
        if self.children:
            for child in self.children.values():
                c_size = child.sort_size
                if c_size > size:
                    size = c_size
        return size

    @property
    def human_size(self) -> str:
        """Convert image size to human-readable value."""
        return self.get_human_size(self.size)

    @property
    def human_net_size(self) -> str:
        """Convert image size to human-readable value."""
        if self.net_size is not None:
            net = self.get_human_size(self.net_size)
            if self.net_size is not None and self.net_size >= 0:
                net = "+" + net
            return net
        else:
            return ''

    def update_net_sizes(self):
        """Update net sizes of chidren images."""
        if not self.children:
            return
        for child in self.children.values():
            child.update_net_sizes()
            if self.size is not None and child.size is not None:
                child.net_size = child.size - self.size

@dataclasses.dataclass
class ColorTheme:
    """Color theme for rendering tree."""

    clear: str = '\u001b[0m'
    image: str = ''
    image_container: str = '\u001b[1m'
    tag: str = ''
    tag_container: str = '\u001b[1m'
    hl: str = '\u001b[1;36m'
    running: str = '\u001b[32m'
    running_end: str = clear
    paused: str = '\u001b[36m'
    paused_end: str = clear
    exited_ko: str = '\u001b[31m'
    exited_ko_end: str = clear
    dead: str = '\u001b[1;31m'
    dead_end: str = clear
    exited_ok: str = '\u001b[30m'
    exited_ok_end: str = clear
    restarting: str = '\u001b[1;33m'
    restarting_end: str = clear
    deleting: str = '\u001b[33m'
    deleting_end: str = clear
    created: str = '\u001b[35m'
    created_end: str = clear
    unknown: str = '\u001b[37m'
    unknown_end: str = clear

    def disable_colors(self):
        """Disable colors."""
        for field in dataclasses.fields(self):
            setattr(self, field.name, '')
        self.running_end = '*'
        self.paused_end = '='
        self.exited_ko_end = self.dead_end = '!'
        self.restarting_end = '?'
        self.deleting_end = '-'
        self.created_end = '+'

    def colorless(self, colored_text: str):
        """Return a colorless version of string with color escape sequences."""
        color_seq = re.compile('\u001b\\[[0-9;]*m')
        return color_seq.sub('', colored_text)


class Docker(ABC):
    """Helper class for docker-related actions.

    This is the base abstract class.
    """

    ImagesData = tuple[dict[str, ImageDict], dict[str, str]]

    @abstractmethod
    def get_images(self) -> ImagesData:
        """Get images list."""

    @abstractmethod
    def get_layers(self, image_id: str) -> list[str]:
        """Get list of layers for given image id."""

    @abstractmethod
    def get_containers(self) -> list[ContainerInfos]:
        """Get containers list."""

    @staticmethod
    def get_helper_instance() -> 'Docker':
        """Get the helper instance based on available interface."""
        if docker_interface == 'package':
            return DockerPackage()
        else:
            logger.warning("docker package not found: using docker command line client (slower).")
            return DockerCli()


class DockerCli(Docker):
    """Helper class for docker-related actions.

    Implementation using the `docker` command line utility.
    """

    docker_exec: list[str] = ['docker']
    sizes_scales: tuple[str] = ('B', 'KB', 'MB', 'GB', 'TB', 'PB')  # PB is already way outside reasonnable range
    size_string_re: re.Pattern = re.compile(r'^([0-9]+(?:\.[0-9]+)?)([KMGTP]?B)$', re.I)

    def size_from_string(self, size_string: str) -> int | None:
        """Convert string representation of an image size to a bytes count."""
        match = self.size_string_re.match(size_string.strip())
        if not match:
            return None
        size = float(match.group(1))
        scale = match.group(2).upper()
        return int(size * (1000 ** self.size_scales.index(scale)))

    def run(self, command_args: Iterable[str], **kwargs) -> subprocess.CompletedProcess:
        """Run the docker command.

        extra keyword arguments are passed in the subprocess.run() function call.
        """
        return subprocess.run(self.docker_exec + list(command_args), **kwargs)

    def get_images(self) -> Docker.ImagesData:
        """Get images list."""
        images = defaultdict(ImageInfos)
        command = ['image', 'ls', '--format', '{{.ID}} {{.Repository}}:{{.Tag}} {{.Size}}']
        called_proc = self.run(command, capture_output=True, encoding='ascii')
        tags_to_images = dict()
        for row in called_proc.stdout.split('\n'):
            row = row.strip()
            if row == "":
                continue
            image_id, tag, size = row.split()[:3]
            if tag == '<none>:<none>':
                tag = '<untagged>'
            else:
                tags_to_images[tag] = image_id
            images[image_id].image_id = image_id
            images[image_id].tags.add(tag)
            images[image_id].size = self.size_from_string(size)
        return dict(images), tags_to_images


    def get_layers(self, image_id: str) -> list[str]:
        """Get list of layers for given image."""
        command = ['inspect', image_id]
        called_proc = self.run(command, capture_output=True, encoding='utf-8', errors='ignore')
        data = json.loads(called_proc.stdout)
        return data[0]['RootFS']['Layers']


    def get_containers(self) -> list[ContainerInfos]:
        """Get containers list."""
        containers = []
        command = ['container', 'ls', '-a', '--format', '{{.ID}};{{.Names}};{{.Image}};{{.Status}}']
        called_proc = self.run(command, capture_output=True, encoding='ascii')
        for row in called_proc.stdout.split('\n'):
            row = row.strip()
            if row == "":
                continue
            id_, names, image, status = row.split(';')[:4]
            container = ContainerInfos(id_, names, image, raw_status=status)
            containers.append(container)
        return containers


class DockerPackage(Docker):
    """Helper class for docker-related acions.

    Implementation using the docker package.
    """

    def __init__(self):
        """Initialize docker client."""
        self._client = docker.from_env()

    def get_images(self) -> Docker.ImagesData:
        """Get images list."""
        images = defaultdict(ImageInfos)
        tags_to_images = dict()
        for image in self._client.images.list():
            short_id = image.short_id.rpartition(':')[2]
            images[short_id].id = short_id
            images[short_id].size = image.attrs.get('Size')
            images[short_id].tags.update(image.tags or ['<untagged>'])
            images[short_id].layers = image.attrs['RootFS']['Layers']
            for tag in image.tags:
                tags_to_images[tag] = short_id
        return dict(images), tags_to_images

    def get_layers(self, image_id: str) -> list[str]:
        """Get list of layers for given image id."""
        image = self._client.images.get(image_id)
        return image.attrs['RootFS']['Layers']

    def get_containers(self) -> list[ContainerInfos]:
        """Get containers list."""
        containers = []
        for container in self._client.containers.list():
            try:
                image_tag = container.image.tags[0]
            except IndexError:
                image_tag = container.image.short_id
            exit_code = container.attrs['State']['ExitCode'] if container.status == 'exited' else None
            status = container.status
            if status == 'exited':
                status += '-ok' if exit_code == 0 else '-ko'
            containers.append(ContainerInfos(container.short_id, container.name, image_tag, status, exit_code))
        return containers


class DockerTree:
    """Generate a tree representation of available docker images tags."""

    docker: Docker

    def __init__(self):
        """Initialize docker tree instance."""
        self.docker = Docker.get_helper_instance()

    def build_tree(self, tag_depth_offset: int) -> tuple[dict[str, ImageDict], int, int]:
        """Build images tree."""
        images, tags_to_images = self.docker.get_images()
        layers_index: dict[str, str] = dict()
        images_children: dict[str, set[str]] = defaultdict(set)
        for image_id in images:
            if not images[image_id].layers:
                images[image_id].layers = self.docker.get_layers(image_id)
            layers_index[','.join(images[image_id].layers)] = image_id
        for container in self.docker.get_containers():
            image_id = tags_to_images.get(container.image_tag)
            if image_id is None and container.image_tag in images:
                image_id = container.image_tag
                container.image_tag = images[image_id].first_tag
            if image_id is not None:
                images[image_id].containers.append(container)
        for image_id, image in images.items():
            image.containers = sorted(image.containers, key=lambda ctn: ctn.id)
            layers = image.layers
            for last_layer in range(len(layers) - 1, 0, -1):
                layer_path = ','.join(layers[:last_layer])
                if layer_path in layers_index:
                    images_children[layers_index[layer_path]].add(image_id)
                    break
        all_children = set()
        for children in images_children.values():
            all_children.update(children)
        tree = dict()
        for image_id in images:
            if image_id in images_children:
                images[image_id].children = dict((child, images[child]) for child in images_children[image_id])
            if image_id not in all_children:
                tree[image_id] = images[image_id]
        for parent in tree.values():
            parent.update_net_sizes()
        _walk = [(0, tree)]
        depth = 0
        max_tag_length = 0
        while _walk:
            c_depth, c_tree = _walk.pop(0)
            depth = max(depth, c_depth)
            for image in c_tree.values():
                c_max_length = max(len(tag) for tag in (image.tags or {''}))
                max_tag_length = max(max_tag_length, c_depth * tag_depth_offset + c_max_length)
            _walk.extend((c_depth + 1, c_tree[k].children) for k in c_tree if c_tree[k].children is not None)
        return tree, depth, max_tag_length


    def _containers_string(self, containers: list[ContainerInfos], tag: str, colors: ColorTheme | None,
                           hl: set[re.Pattern] | None = None):
        """Build a comma separated list of containers with given tag, with colors applied."""
        filtered = []
        for container in containers:
            if container.image_tag != tag:
                continue
            if colors:
                status_key = container.status.replace('-', '_')
                if hl:
                    for pat in hl:
                        if pat.search(container.name):
                            status_start = colors.hl
                            status_end = colors.clear
                            break
                    else:
                        status_start = ''
                        status_end = ''
                else:
                    status_start = getattr(colors, status_key, '')
                    status_end = getattr(colors, f'{status_key}_end', '')
            else:
                status_start = status_end = ''
            filtered.append(f"{status_start}{container.name}{status_end}")
        return ", ".join(filtered)


    def _add_line(self, lines: list[str], colors: ColorTheme | None, prefix: str, mark: str, image_id: str, size: str,
                  net_size: str, tag: str, containers: list[ContainerInfos] | str, max_tag_length: int, max_depth: int,
                  depth: int = 0, hl: set[re.Pattern] | None = None):
        if isinstance(containers, str):
            tag_containers = containers
        else:
            tag_containers = self._containers_string(containers, tag, colors, hl)
            #tag_containers = ', '.join(ctn.name + ('*' if ctn.is_running() else '')
                                       #for ctn in containers if ctn.image_tag == tag)
        offset = (' ' * (len(mark) + 1)) * (max_depth - depth)
        tag_offset = ' ' * (max_tag_length - len(tag) - (len(mark) + 1) * depth)
        if colors:
            c_image = colors.image_container if containers else colors.image
            c_tag = colors.tag_container if containers else colors.tag
            if hl:
                c_image = colors.image
                c_tag = colors.tag
                for pat in hl:
                    if pat.search(image_id):
                        c_image = colors.hl
                    if pat.search(tag):
                        c_tag = colors.hl
            else:
                c_image = colors.image_container if containers else colors.image
                c_tag = colors.tag_container if containers else colors.tag
            c_clear = colors.clear
        else:
            c_image = c_tag = c_clear = ''
        if net_size:
            net_size = f"({net_size})"
        lines.append(f"{prefix}{mark} {c_image}{image_id:12}{c_clear}{offset}    {size:6} {net_size:9}   "
                     f"{prefix}{mark} {c_tag}{tag}{c_clear}    {tag_offset}{tag_containers}")


    def collect_lines(self, images: dict[str, ImageDict] | None, colors: ColorTheme | None,
                      max_tag_length: int = 0, max_depth: int = 0, depth: int = 0, prefix: str = '',
                      hl: set[re.Pattern] | None = None, sort: str = 'tags', reverse: bool = False) -> list[str]:
        """Collect infos lines."""
        if images is None:
            return []
        lines = []
        count = len(images)
        if sort == 'size':
            images_iter = sorted(images.items(), key=lambda item: item[1].sort_size, reverse=reverse)
        else:
            images_iter = sorted(images.items(), key=lambda item: min(item[1].tags or {''}), reverse=reverse)

        for index, (image_id, data) in enumerate(images_iter, start=1):
            tags = data.sorted_tags(reverse=reverse)
            is_last = index == count
            mark = ('\u250c' if (index == 1 and depth == 0) else ('\u2514' if is_last else '\u251C')) + ('\u2500' * 2)
            tag_prefix = '\u2502  ' if not is_last else '   '
            self._add_line(lines, colors, prefix, mark, image_id[:12], data.human_size, data.human_net_size, tags[0],
                           data.containers, max_tag_length, max_depth, depth, hl=hl)
            for tag in tags[1:]:
                children_mark = '\u2502' if data.children else ''
                self._add_line(lines, colors, prefix, tag_prefix, children_mark, '', '', tag, data.containers,
                               max_tag_length, max_depth, depth, hl=hl)
            nested_prefix = prefix + ('    ' if is_last else '\u2502   ')
            children_lines = self.collect_lines(data.children, colors, max_tag_length, max_depth, depth + 1,
                                                nested_prefix, hl=hl, sort=sort, reverse=reverse)
            lines.extend(children_lines)
        return lines


    def print_tree(self, highlights: set[re.Pattern] | None = None, sort: str = 'tags', reverse: bool = False):
        """Print infos for given images dict."""
        images, max_depth, max_tag_length = self.build_tree(4)
        colors = ColorTheme()
        if os.getenv('USECOLOR', 'no')[:1].lower() not in ('y', '1'):
            colors.disable_colors()
        head_lines = []
        self._add_line(head_lines, None, '', '   ', 'Image Id', 'Size', 'Net', 'Tags', 'Containers',
                       max_tag_length, max_depth)
        lines = self.collect_lines(images, colors, max_tag_length, max_depth, hl=highlights, sort=sort, reverse=reverse)
        delim_line = "-" * max(len(colors.colorless(line)) for line in head_lines + lines)
        head_lines.append(delim_line)
        for line in head_lines + lines:
            print(line)
        if not highlights:
            print(delim_line)
            legend = []
            for status in ContainerInfos.statuses:
                status_key = status.replace('-', '_')
                c_start = getattr(colors, status_key, '')
                c_end = getattr(colors, f'{status_key}_end', '')
                legend.append(f"{c_start}{status}{c_end}")
            print('Containers legend: ', "  ".join(legend))


def _main_cli():
    """Parse command line and print images tree."""
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('highlights', nargs='*',
                        help="Regular expressions to highlight matching ids, tags or container names.")
    parser.add_argument('--size', '-s', dest='sort', action='store_const', const='size', default='tags',
                        help="Sort images by size")
    parser.add_argument('--reverse', '-r', dest='reverse', action='store_true',
                        help="Reverse images sort order")
    options = parser.parse_args()
    highlights = set(re.compile(pat, re.I) for pat in options.highlights)
    docker_tree = DockerTree()
    docker_tree.print_tree(highlights, options.sort, options.reverse)


if __name__ == '__main__':
    _main_cli()
