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
HAS_ARM=$(rustup target list --installed 2>/dev/null | grep -c "^${ARM_TARGET}$" || true)
HAS_X86=$(rustup target list --installed 2>/dev/null | grep -c "^${X86_TARGET}$" || true)

for entry in "${PREPROCESSORS[@]}"; do
    dir="${entry%%:*}"
    bin="${entry##*:}"

    echo "Building preprocessor: $bin"
    pushd "$dir" > /dev/null

    if [[ "$HAS_ARM" -gt 0 && "$HAS_X86" -gt 0 ]]; then
        MACOSX_DEPLOYMENT_TARGET=11.0  cargo build --release --target "$ARM_TARGET"
        MACOSX_DEPLOYMENT_TARGET=10.15 cargo build --release --target "$X86_TARGET"
        lipo -create -output "target/release/$bin" \
            "target/$ARM_TARGET/release/$bin" \
            "target/$X86_TARGET/release/$bin"
        tarball="${bin}-darwin-universal.tar.gz"
        tar -czf "$tarball" -C target/release "$bin"
        gh release upload "v$VERSION" "$tarball" --clobber
        rm -f "$tarball"
    elif [[ "$HAS_ARM" -gt 0 ]]; then
        MACOSX_DEPLOYMENT_TARGET=11.0 cargo build --release --target "$ARM_TARGET"
        cp "target/$ARM_TARGET/release/$bin" "target/release/$bin"
        tarball="${bin}-darwin-arm64.tar.gz"
        tar -czf "$tarball" -C target/release "$bin"
        gh release upload "v$VERSION" "$tarball" --clobber
        rm -f "$tarball"
    else
        MACOSX_DEPLOYMENT_TARGET=10.15 cargo build --release --target "$X86_TARGET"
        cp "target/$X86_TARGET/release/$bin" "target/release/$bin"
        tarball="${bin}-darwin-x86_64.tar.gz"
        tar -czf "$tarball" -C target/release "$bin"
        gh release upload "v$VERSION" "$tarball" --clobber
        rm -f "$tarball"
    fi

    echo "Uploaded $bin to v$VERSION"
    popd > /dev/null
done
