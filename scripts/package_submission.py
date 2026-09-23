"""Package unchanged local numerical sources and reviewer-reproduction files."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = ("results", "derived_results")
REVIEW_DIRS = DATA_DIRS + ("src", "configs", "scripts", "manuscript")
MAP_FILES = ("README.md", "SOURCE_DATA_README.md", "source_data/README.md", "requirements.txt")


def selected(directories, include_figures=False):
    paths = [ROOT / name for name in MAP_FILES]
    for directory in directories:
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix != ".pyc")
    if include_figures:
        paths.extend(p for p in (ROOT / "figures" / "results").glob("*")
                     if p.suffix in (".pdf", ".svg"))
    return sorted(set(paths), key=lambda p: p.relative_to(ROOT).as_posix())


def package(name, paths):
    target = ROOT / name
    with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with ZipFile(target) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(paths)
        assert all(archive.read(path.relative_to(ROOT).as_posix()) == path.read_bytes()
                   for path in paths)
    print(f"{name}: {len(paths)} byte-verified entries ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    package("source_data.zip", selected(DATA_DIRS))
    package("reviewer_data_code.zip", selected(REVIEW_DIRS, include_figures=True))
