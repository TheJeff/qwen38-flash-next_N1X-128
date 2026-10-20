# Copy the downloaded model from /mnt/c (NTFS) onto WSL ext4 and fix ownership.
# 9p (the /mnt/c mount) breaks O_DIRECT and is slow — the engine needs ext4.
set -euo pipefail
SRC="${1:?usage: install-model.sh <windows-src-dir> [dest-name]}"
NAME="${2:-$(basename "$SRC")}"
mkdir -p "$HOME/models/$NAME"
cp -r "$SRC"/. "$HOME/models/$NAME"/
sudo chown -R "$USER":"$USER" "$HOME/models/$NAME"
echo "installed at $HOME/models/$NAME: $(du -sh "$HOME/models/$NAME" | cut -f1)"
