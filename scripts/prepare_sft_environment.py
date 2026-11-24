"""Reuse compatible Torch files without inheriting unrelated serving packages."""

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import venv

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def read_requirements(path):
    requirements = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("-r "):
            requirements.update(read_requirements(path.parent / line[3:]))
        elif line and not line.startswith("#"):
            requirement = Requirement(line)
            requirements[canonicalize_name(requirement.name)] = requirement
    return requirements


def shared_distributions(requirements):
    selected = [requirements[name] for name in ("torch", "torchvision", "triton")]
    if metadata.version("torch") in requirements["torch"].specifier:
        selected.extend(Requirement(value) for value in metadata.requires("torch")
                        if canonicalize_name(Requirement(value).name).startswith("nvidia-"))
    for requirement in selected:
        try:
            distribution = metadata.distribution(requirement.name)
        except metadata.PackageNotFoundError:
            continue
        applies = requirement.marker is None or requirement.marker.evaluate()
        if applies and distribution.version in requirement.specifier:
            yield distribution


def link_distribution(distribution, site):
    roots = set()
    for file in distribution.files or ():
        parts = file.parts
        if parts[0] != "..":
            roots.add(Path(*parts[:2]) if parts[0] == "nvidia" else Path(parts[0]))
    for root in roots:
        source = Path(distribution.locate_file(root)).resolve()
        target = site / root
        if source.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=source.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", type=Path)
    parser.add_argument("requirements", type=Path)
    args = parser.parse_args()
    venv.EnvBuilder(with_pip=True, system_site_packages=False, symlinks=True).create(args.environment)
    python = args.environment / "bin/python"
    site = Path(subprocess.check_output(
        [str(python), "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"], text=True).strip())
    (site / "serving-torch.pth").unlink(missing_ok=True)
    shared = {}
    for distribution in shared_distributions(read_requirements(args.requirements)):
        link_distribution(distribution, site)
        shared[canonicalize_name(distribution.metadata["Name"])] = distribution.version
    output = args.environment.parent
    (output / "shared-torch-packages.json").write_text(json.dumps(shared, indent=2) + "\n")
    plan = output / "environment-plan.json"
    subprocess.run([
        str(python), "-m", "pip", "install", "--dry-run", "--no-compile", "--report", str(plan),
        "--index-url", os.environ.get("DRAWIO_PYPI_INDEX", "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"),
        "-r", str(args.requirements),
    ], check=True)
    changes = {canonicalize_name(item["metadata"]["name"])
               for item in json.loads(plan.read_text())["install"]}
    if changes.intersection(shared):
        raise RuntimeError("Installation would replace a shared Torch dependency")


if __name__ == "__main__":
    main()
