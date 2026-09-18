#!/usr/bin/env sh
set -eu

url='https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz'
size=$(curl -fsSI "$url" | tr -d '\r' | awk 'tolower($1) == "content-length:" { print $2; exit }')
case "$size" in
  ''|*[!0-9]*) echo 'could not determine ffmpeg archive size' >&2; exit 1 ;;
esac
# ponytail: source throttles one connection; use eight bounded ranges for this one-time server setup.
parts=8
workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT
pids=''

for index in $(seq 0 $((parts - 1))); do
  start=$((size * index / parts))
  end=$((size * (index + 1) / parts - 1))
  curl -fL --retry 3 --connect-timeout 15 -r "$start-$end" -o "$workdir/$index.part" "$url" &
  pids="$pids $!"
done
for pid in $pids; do
  wait "$pid"
done

for index in $(seq 0 $((parts - 1))); do
  cat "$workdir/$index.part"
done > "$workdir/ffmpeg.tar.xz"

tar -xJf "$workdir/ffmpeg.tar.xz" -C "$workdir"
ffmpeg_bin=$(find "$workdir" -type f -name ffmpeg -perm -111 | head -n 1)
test -n "$ffmpeg_bin"
sudo install -d -m 755 /opt/good-badminton/bin
sudo install -m 755 "$ffmpeg_bin" /opt/good-badminton/bin/ffmpeg
sha256sum /opt/good-badminton/bin/ffmpeg
/opt/good-badminton/bin/ffmpeg -version | head -n 1
