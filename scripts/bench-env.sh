#!/usr/bin/env bash
# ^ Shared local state setup for benchmark scripts.

bench_env_init() {
    local repo_root="$1"
    local scope="$2"
    local source_xdg="${XDG_CONFIG_HOME:-$HOME/.config}"
    local state_root="$repo_root/.bench-state/$scope"
    export XDG_CONFIG_HOME="$state_root/xdg"
    export TMPDIR="$state_root/tmp"
    mkdir -p "$XDG_CONFIG_HOME/ir" "$TMPDIR"

    local source_config="$source_xdg/ir/config.yml"
    local local_config="$XDG_CONFIG_HOME/ir/config.yml"
    if [[ -f "$source_config" && ! -f "$local_config" ]]; then
        cp "$source_config" "$local_config"
    fi
}
