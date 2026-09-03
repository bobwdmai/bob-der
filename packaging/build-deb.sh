#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUTPUT_DIR=${1:-"$PROJECT_ROOT/dist"}
VERSION=$(python3 -c 'import pathlib,sys,tomllib; print(tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["project"]["version"])' "$PROJECT_ROOT/pyproject.toml")
BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/bob-der-deb.XXXXXXXX")
PACKAGE_ROOT="$BUILD_ROOT/root"

cleanup() {
    case "$BUILD_ROOT" in
        "${TMPDIR:-/tmp}"/bob-der-deb.*) rm -rf -- "$BUILD_ROOT" ;;
    esac
}
trap cleanup EXIT HUP INT TERM

mkdir -p \
    "$PACKAGE_ROOT/DEBIAN" \
    "$PACKAGE_ROOT/usr/bin" \
    "$PACKAGE_ROOT/usr/lib/bob-der" \
    "$PACKAGE_ROOT/usr/share/doc/bob-der" \
    "$PACKAGE_ROOT/usr/share/man/man1" \
    "$OUTPUT_DIR"

python3 -m pip install \
    --quiet \
    --disable-pip-version-check \
    --no-cache-dir \
    --no-compile \
    --target "$PACKAGE_ROOT/usr/lib/bob-der" \
    "$PROJECT_ROOT"

# pip --target creates a redundant entry-point directory; Debian owns /usr/bin.
rm -rf -- "$PACKAGE_ROOT/usr/lib/bob-der/bin"
find "$PACKAGE_ROOT/usr/lib/bob-der" -type d -name __pycache__ \
    -prune -exec rm -rf -- {} +
find "$PACKAGE_ROOT/usr/lib/bob-der" -type d \
    \( -name tests -o -name test \) \
    -prune -exec rm -rf -- {} +
find "$PACKAGE_ROOT/usr/lib/bob-der" -type f \
    \( -name '*.pyc' -o -name '*.pyo' \) -delete
find "$PACKAGE_ROOT/usr/lib/bob-der" -type f -name direct_url.json -delete

# Drop optional CPython accelerators only when an equivalent pure-Python module
# is present. This keeps the resulting package usable across Python 3.10+.
ARCHITECTURE=all
find "$PACKAGE_ROOT/usr/lib/bob-der" -type f -name '*.so' -print |
while IFS= read -r extension; do
    module=${extension%%.cpython-*}
    if [ -f "$module.py" ]; then
        rm -f -- "$extension"
    else
        printf '%s\n' "$extension" >> "$BUILD_ROOT/native-extensions"
    fi
done
if [ -s "$BUILD_ROOT/native-extensions" ]; then
    ARCHITECTURE=$(dpkg --print-architecture)
fi

install -m 0755 "$PROJECT_ROOT/packaging/debian/bob-der" \
    "$PACKAGE_ROOT/usr/bin/bob-der"
install -m 0755 "$PROJECT_ROOT/packaging/debian/bob-der-setup" \
    "$PACKAGE_ROOT/usr/bin/bob-der-setup"
install -m 0644 "$PROJECT_ROOT/README.md" \
    "$PACKAGE_ROOT/usr/share/doc/bob-der/README.md"
install -m 0644 "$PROJECT_ROOT/packaging/debian/copyright" \
    "$PACKAGE_ROOT/usr/share/doc/bob-der/copyright"
sed "s/@VERSION@/$VERSION/g" "$PROJECT_ROOT/packaging/debian/changelog" |
    gzip -n -9 > "$PACKAGE_ROOT/usr/share/doc/bob-der/changelog.gz"
sed "s/@VERSION@/$VERSION/g" "$PROJECT_ROOT/packaging/debian/bob-der.1" |
    gzip -n -9 > "$PACKAGE_ROOT/usr/share/man/man1/bob-der.1.gz"
chmod 0644 "$PACKAGE_ROOT/usr/share/man/man1/bob-der.1.gz"

# Normalize modes regardless of the builder's umask. Only the public launcher
# is executable; imported Python files and package metadata are regular files.
find "$PACKAGE_ROOT" -type d -exec chmod 0755 {} +
find "$PACKAGE_ROOT" -type f -exec chmod 0644 {} +
chmod 0755 "$PACKAGE_ROOT/usr/bin/bob-der"
chmod 0755 "$PACKAGE_ROOT/usr/bin/bob-der-setup"

INSTALLED_SIZE=$(du -sk "$PACKAGE_ROOT/usr" | awk '{print $1}')
sed \
    -e "s/@VERSION@/$VERSION/g" \
    -e "s/@ARCHITECTURE@/$ARCHITECTURE/g" \
    -e "s/@INSTALLED_SIZE@/$INSTALLED_SIZE/g" \
    "$PROJECT_ROOT/packaging/debian/control.in" > "$PACKAGE_ROOT/DEBIAN/control"
chmod 0644 "$PACKAGE_ROOT/DEBIAN/control"

PACKAGE_PATH="$OUTPUT_DIR/bob-der_${VERSION}_${ARCHITECTURE}.deb"
# gzip is understood by older graphical package installers that may label
# newer zstd-compressed Debian packages as generic "Data" files.
dpkg-deb --root-owner-group -Zgzip -z9 --build "$PACKAGE_ROOT" "$PACKAGE_PATH"
(
    cd "$OUTPUT_DIR"
    sha256sum "$(basename "$PACKAGE_PATH")" > "$(basename "$PACKAGE_PATH").sha256"
)
printf '%s\n' "$PACKAGE_PATH"
