import csv
import json
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
WIDTH = 1280
HEIGHT = 720
FPS = 30
DURATION_SEC = 10.0
COURT_X1 = 70
COURT_X2 = 930
COURT_Y1 = 145
COURT_Y2 = 536
COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_events(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def court_to_screen(x_m, y_m):
    sx = COURT_X1 + (float(y_m) / COURT_LENGTH_M) * (COURT_X2 - COURT_X1)
    sy = COURT_Y1 + (float(x_m) / COURT_WIDTH_M) * (COURT_Y2 - COURT_Y1)
    return int(round(sx)), int(round(sy))


def interpolate(rows, field, time_sec):
    times = np.asarray([float(row["time_sec"]) for row in rows], dtype=np.float64)
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return float(np.interp(time_sec, times, values))


def nearest_row(rows, time_sec):
    index = min(range(len(rows)), key=lambda idx: abs(float(rows[idx]["time_sec"]) - time_sec))
    return rows[index]


def draw_text(frame, text, origin, scale=0.58, color=(230, 235, 240), thickness=1):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_court(frame):
    cv2.rectangle(frame, (COURT_X1, COURT_Y1), (COURT_X2, COURT_Y2), (37, 112, 69), -1)
    line = (232, 236, 225)
    cv2.rectangle(frame, (COURT_X1, COURT_Y1), (COURT_X2, COURT_Y2), line, 3)

    for y_m in (0.76, 4.72, 6.70, 8.68, 12.64):
        x, _ = court_to_screen(0, y_m)
        cv2.line(frame, (x, COURT_Y1), (x, COURT_Y2), line, 2)

    for x_m in (0.46, COURT_WIDTH_M / 2.0, COURT_WIDTH_M - 0.46):
        _, y = court_to_screen(x_m, 0)
        if x_m == COURT_WIDTH_M / 2.0:
            x_start, _ = court_to_screen(0, 0)
            x_end, _ = court_to_screen(0, 4.72)
            cv2.line(frame, (x_start, y), (x_end, y), line, 2)
            x_start, _ = court_to_screen(0, 8.68)
            x_end, _ = court_to_screen(0, 13.4)
            cv2.line(frame, (x_start, y), (x_end, y), line, 2)
        else:
            cv2.line(frame, (COURT_X1, y), (COURT_X2, y), line, 2)

    net_x, _ = court_to_screen(0, 6.70)
    cv2.line(frame, (net_x, COURT_Y1 - 12), (net_x, COURT_Y2 + 12), (190, 200, 210), 5)
    cv2.line(frame, (net_x, COURT_Y1 - 12), (net_x, COURT_Y2 + 12), (55, 60, 67), 1)
    draw_text(frame, "PLAYER B", (COURT_X1 + 12, COURT_Y1 - 22), 0.55, (101, 189, 255), 2)
    draw_text(frame, "PLAYER A", (COURT_X2 - 112, COURT_Y1 - 22), 0.55, (255, 169, 82), 2)


def draw_stick_player(frame, position, orientation_deg, color, label, is_hitting=False, predicted=False):
    center = np.asarray(position, dtype=np.float64)
    theta = math.radians(orientation_deg)
    court_vector = np.asarray([math.cos(theta), math.sin(theta)])
    screen_forward = np.asarray([court_vector[1], court_vector[0]])
    norm = np.linalg.norm(screen_forward)
    if norm < 1e-6:
        screen_forward = np.asarray([1.0, 0.0])
    else:
        screen_forward /= norm
    side = np.asarray([-screen_forward[1], screen_forward[0]])

    hip = center
    chest = center + screen_forward * 13
    head = center + screen_forward * 25
    shoulder_left = chest + side * 8
    shoulder_right = chest - side * 8
    reach = 22 if is_hitting else 14
    hand_left = shoulder_left + screen_forward * (reach * 0.55) + side * (reach * 0.35)
    hand_right = shoulder_right + screen_forward * reach - side * (reach * 0.15)
    foot_left = hip - screen_forward * 15 + side * 10
    foot_right = hip - screen_forward * 15 - side * 10

    thickness = 3
    if predicted:
        color = tuple(int(channel * 0.62) for channel in color)
    cv2.line(frame, tuple(hip.astype(int)), tuple(chest.astype(int)), color, thickness, cv2.LINE_AA)
    cv2.line(frame, tuple(shoulder_left.astype(int)), tuple(hand_left.astype(int)), color, thickness, cv2.LINE_AA)
    cv2.line(frame, tuple(shoulder_right.astype(int)), tuple(hand_right.astype(int)), color, thickness, cv2.LINE_AA)
    cv2.line(frame, tuple(hip.astype(int)), tuple(foot_left.astype(int)), color, thickness, cv2.LINE_AA)
    cv2.line(frame, tuple(hip.astype(int)), tuple(foot_right.astype(int)), color, thickness, cv2.LINE_AA)
    cv2.circle(frame, tuple(head.astype(int)), 7, color, -1, cv2.LINE_AA)
    cv2.circle(frame, tuple(center.astype(int)), 28, color, 1, cv2.LINE_AA)
    draw_text(frame, label + (" (pred)" if predicted else ""), (int(center[0]) - 30, int(center[1]) + 44), 0.43, color, 1)


def draw_ball(frame, x_m, y_m, z_m, predicted=False):
    shadow = np.asarray(court_to_screen(x_m, y_m), dtype=np.int32)
    lift = int(min(42, max(4, z_m * 7)))
    ball = shadow + np.asarray([0, -lift], dtype=np.int32)
    color = (120, 205, 255) if not predicted else (85, 125, 150)
    radius = int(min(8, 3 + z_m * 0.55))
    cv2.ellipse(frame, tuple(shadow), (8, 3), 0, 0, 360, (35, 65, 45), -1, cv2.LINE_AA)
    cv2.line(frame, tuple(shadow), tuple(ball), (150, 170, 155), 1, cv2.LINE_AA)
    cv2.circle(frame, tuple(ball), radius + 3, (40, 70, 80), -1, cv2.LINE_AA)
    cv2.circle(frame, tuple(ball), radius, color, -1, cv2.LINE_AA)
    draw_text(frame, f"{z_m:.1f}m", (int(ball[0]) + 9, int(ball[1]) - 5), 0.38, (235, 239, 225), 1)


def current_event(events, time_sec):
    active = events[0]
    for event in events:
        if float(event["time_sec"]) <= time_sec:
            active = event
        else:
            break
    return active


def draw_side_panel(frame, time_sec, player_row, ball_row, event):
    x = 985
    draw_text(frame, "SIMULATED RALLY", (x, 88), 0.86, (242, 244, 248), 2)
    draw_text(frame, f"Time  {time_sec:05.2f} s", (x, 135), 0.72, (205, 213, 224), 2)
    draw_text(frame, f"Shot  {event['shot_id']:02d} / 12", (x, 180), 0.62, (205, 213, 224), 1)
    draw_text(frame, event["stroke_type"].replace("_", " ").upper(), (x, 218), 0.59, (120, 205, 255), 2)
    draw_text(frame, f"Hitter: {event['hitter_id']}", (x, 251), 0.50, (215, 221, 231), 1)

    draw_text(frame, "PLAYER MOTION", (x, 315), 0.54, (168, 178, 192), 1)
    draw_text(frame, f"A speed  {float(player_row['player_a_speed_mps']):4.1f} m/s", (x, 351), 0.49, (255, 169, 82), 1)
    draw_text(frame, f"A zone   {player_row['player_a_zone']}", (x, 380), 0.44, (225, 229, 235), 1)
    draw_text(frame, f"B speed  {float(player_row['player_b_speed_mps']):4.1f} m/s", (x, 421), 0.49, (101, 189, 255), 1)
    draw_text(frame, f"B zone   {player_row['player_b_zone']}", (x, 450), 0.44, (225, 229, 235), 1)

    draw_text(frame, "SHUTTLE", (x, 512), 0.54, (168, 178, 192), 1)
    draw_text(frame, f"Height   {float(ball_row['z_m']):4.1f} m", (x, 548), 0.48, (225, 229, 235), 1)
    draw_text(frame, f"Status   {ball_row['tracking_status']}", (x, 577), 0.46, (225, 229, 235), 1)

    if event.get("result") != "in_play" and time_sec >= float(event["time_sec"]):
        draw_text(frame, "NET ERROR", (x, 635), 0.75, (95, 95, 245), 2)
        draw_text(frame, "Winner: player_a", (x, 671), 0.53, (255, 169, 82), 2)


def render(output_path):
    players = read_csv(HERE / "player_tracking_10hz.csv")
    shuttle = read_csv(HERE / "shuttlecock_tracking_30hz.csv")
    events = read_events(HERE / "events.jsonl")
    temp_path = output_path.with_name(output_path.stem + ".mp4v.tmp.mp4")
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Unable to create temporary MP4 video")

    for frame_index in range(int(DURATION_SEC * FPS) + 1):
        time_sec = frame_index / FPS
        player_row = nearest_row(players, time_sec)
        ball_row = nearest_row(shuttle, time_sec)
        event = current_event(events, time_sec)
        frame = np.full((HEIGHT, WIDTH, 3), (22, 25, 31), dtype=np.uint8)
        draw_court(frame)

        ax = interpolate(players, "player_a_x_m", time_sec)
        ay = interpolate(players, "player_a_y_m", time_sec)
        bx = interpolate(players, "player_b_x_m", time_sec)
        by = interpolate(players, "player_b_y_m", time_sec)
        a_orientation = interpolate(players, "player_a_orientation_deg", time_sec)
        b_orientation = interpolate(players, "player_b_orientation_deg", time_sec)
        hit_window = abs(time_sec - float(event["time_sec"])) <= 0.12
        draw_stick_player(
            frame,
            court_to_screen(ax, ay),
            a_orientation,
            (255, 169, 82),
            "A",
            hit_window and event["hitter_id"] == "player_a",
            player_row["player_a_tracking_status"] == "predicted",
        )
        draw_stick_player(
            frame,
            court_to_screen(bx, by),
            b_orientation,
            (101, 189, 255),
            "B",
            hit_window and event["hitter_id"] == "player_b",
            player_row["player_b_tracking_status"] == "predicted",
        )
        draw_ball(
            frame,
            float(ball_row["x_m"]),
            float(ball_row["y_m"]),
            float(ball_row["z_m"]),
            ball_row["tracking_status"] == "predicted",
        )
        draw_side_panel(frame, time_sec, player_row, ball_row, event)
        writer.write(frame)

    writer.release()

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(temp_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1200:])
    os.remove(temp_path)
    return output_path


if __name__ == "__main__":
    rendered = render(HERE / "stick_match_10s.mp4")
    print(rendered)
