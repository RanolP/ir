#!/bin/bash
set -euo pipefail
# Called by rust-release after main GitHub release.
# Env: VERSION, BINARY_NAME, GITHUB_REPO, ARCH_LABEL

# Preprocessor crates: (subdir, binary_name)
PREPROCESSORS=(
    "preprocessors/ko/lindera-tokenize:lindera-tokenize"
    "preprocessors/ja/lindera-tokenize:lindera-tokenize-ja"
    "preprocessors/zh/bigram-tokenize:bigram-tokenize-zh"
)

ARM_TARGET="aarch64-apple-darwin"
X86_TARGET="x86_64-apple-darwin"

LINUX_TARGETS=(
    "x86_64-unknown-linux-musl:linux-x86_64"
    "aarch64-unknown-linux-musl:linux-aarch64"
)

for target in "$ARM_TARGET" "$X86_TARGET"; do
    if ! rustup target list --installed 2>/dev/null | grep -q "^${target}$"; then
        echo "ERROR: required target '$target' not installed. Run: rustup target add $target"
        exit 1
    fi
done

# macOS universal binaries
for entry in "${PREPROCESSORS[@]}"; do
    dir="${entry%%:*}"
    bin="${entry##*:}"

    echo "Building macOS universal preprocessor: $bin"
    pushd "$dir" > /dev/null

    MACOSX_DEPLOYMENT_TARGET=11.0  cargo build --release --target "$ARM_TARGET"
    MACOSX_DEPLOYMENT_TARGET=10.15 cargo build --release --target "$X86_TARGET"
    lipo -create -output "target/release/$bin" \
        "target/$ARM_TARGET/release/$bin" \
        "target/$X86_TARGET/release/$bin"
    tarball="${bin}-darwin-universal.tar.gz"
    tar -czf "$tarball" -C target/release "$bin"
    gh release upload "v$VERSION" "$tarball" --clobber
    rm -f "$tarball"

    echo "Uploaded $bin (darwin-universal) to v$VERSION"
    popd > /dev/null
done

# Linux static binaries (requires cross: cargo install cross --git https://github.com/cross-rs/cross)
if command -v cross &>/dev/null; then
    for entry in "${PREPROCESSORS[@]}"; do
        dir="${entry%%:*}"
        bin="${entry##*:}"

        echo "Building Linux preprocessors: $bin"
        pushd "$dir" > /dev/null

        for target_entry in "${LINUX_TARGETS[@]}"; do
            target="${target_entry%%:*}"
            label="${target_entry##*:}"

            cross build --release --target "$target"
            tarball="${bin}-${label}.tar.gz"
            tar -czf "$tarball" -C "target/$target/release" "$bin"
            gh release upload "v$VERSION" "$tarball" --clobber
            rm -f "$tarball"
            echo "Uploaded $bin ($label) to v$VERSION"
        done

        popd > /dev/null
    done
else
    echo "INFO: 'cross' not found — skipping Linux preprocessor builds."
    echo "      Install with: cargo install cross --git https://github.com/cross-rs/cross"
fi
