#!/usr/bin/env bash
# Kopiuje pakiety do kontenera ARM_05, w razie potrzeby pobiera explore_lite
# (m-explore-ros2) i buduje wszystko w /arm_ws (ROS 2 Jazzy).
#   ./deploy.sh            # kopiowanie + budowanie
#   ./deploy.sh --test     # dodatkowo testy jednostkowe
set -euo pipefail

CONTAINER=${CONTAINER:-ARM_05}
HERE=$(cd "$(dirname "$0")" && pwd)
EXPLORE_REPO=https://github.com/robo-friends/m-explore-ros2.git
EXPLORE_COMMIT=326cf8a0b487c34246bb8f3326afbcd69576dc60

if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != "true" ]; then
    echo "Kontener $CONTAINER nie dziala (docker start $CONTAINER)." >&2
    exit 1
fi

docker exec "$CONTAINER" mkdir -p /arm_ws/src/lab5
docker cp "$HERE/src/." "$CONTAINER":/arm_ws/src/lab5/

docker exec "$CONTAINER" bash -c "
set -e
cd /arm_ws/src
if [ ! -d m-explore-ros2 ]; then
    git clone $EXPLORE_REPO m-explore-ros2
    git -C m-explore-ros2 checkout -q $EXPLORE_COMMIT
fi
touch m-explore-ros2/map_merge/COLCON_IGNORE   # pakiet map_merge nie jest potrzebny
source /opt/ros/jazzy/setup.bash
cd /arm_ws
colcon build --symlink-install --packages-up-to cube_search explore_lite
"

if [ "${1:-}" = "--test" ]; then
    docker exec "$CONTAINER" bash -c "
source /opt/ros/jazzy/setup.bash && source /arm_ws/install/setup.bash
cd /arm_ws/src/lab5/cube_search && python3 -m pytest -q test
"
fi
echo "Gotowe. Uruchomienie (w kontenerze, export ROS_DOMAIN_ID=67):"
echo "  ros2 launch cube_search cube_search.launch.py"
echo "  ros2 run cube_search search_client"
