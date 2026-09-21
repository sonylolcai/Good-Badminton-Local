import argparse
import os

from badminton_analysis.system import BadmintonAnalysisSystem, load_runtime_dependencies


def normalized_roi(value):
    try:
        roi = tuple(float(item.strip()) for item in value.split(','))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('ROI 必须是 x1,y1,x2,y2 四个归一化数值') from exc
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise argparse.ArgumentTypeError('ROI 必须满足 0 <= x1 < x2 <= 1 且 0 <= y1 < y2 <= 1')
    return roi


def image_line(value):
    try:
        values = tuple(float(item.strip()) for item in value.split(','))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('球网必须是 x1,y1,x2,y2 四个图像像素值') from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError('球网必须是 x1,y1,x2,y2 四个图像像素值')
    return [(values[0], values[1]), (values[2], values[3])]



def main():
    parser = argparse.ArgumentParser(description='羽毛球比赛视频分析系统')
    parser.add_argument('--video-path', default='videos/demo.mp4', type=str, help='输入视频文件路径')
    parser.add_argument('--template-path', default='templates/demo.png', type=str, help='球场模板图像路径；不提供时会弹出文件选择框')
    parser.add_argument('--output-dir', default=None, type=str, help='输出目录，默认 outputs/<视频文件名>')
    parser.add_argument('--ball-model', default='weights/yolo11s-ball.pt', type=str, help='YOLO 羽毛球检测模型路径')
    parser.add_argument('--pose-family', default='yolo-pose', choices=['rtmpose', 'rtmo', 'yolo-pose'], help='姿态模型族')
    parser.add_argument('--pose-mode', default='balanced', choices=['lightweight', 'balanced', 'performance'], help='RTMPose / RTMO 模型档位')
    parser.add_argument('--yolo-pose-model', default='weights/yolo11n-pose.pt', type=str, help='YOLO pose 模型路径或模型名')
    parser.add_argument('--pose-imgsz', default=1280, type=int, choices=[640, 960, 1280], help='YOLO Pose 全画面推理尺寸，当前固定机位基线推荐1280')
    parser.add_argument('--pose-sample-hz', default=0.0, type=float, help='姿态采样频率；0 表示每个源视频帧均推理并保存 17 个关节，正数为降采样')
    parser.add_argument('--pose-conf', default=0.15, type=float, help='YOLO Pose 人体置信阈值，固定低清机位默认0.15')
    parser.add_argument('--far-player-enhancement', choices=['true', 'false'], default='false', help='启用全场640加远端ROI二次640检测，默认关闭')
    parser.add_argument('--far-pose-roi', type=normalized_roi, default=(0.12, 0.30, 0.86, 0.82), help='相对于姿态ROI的远端检测区域 x1,y1,x2,y2')
    parser.add_argument('--net-image-line', type=image_line, default=None, help='人工球网两端像素坐标 x1,y1,x2,y2；不传时由球场四角推导')
    parser.add_argument('--pose-roi', choices=['true', 'false'], default='true', help='是否显示姿态检测 ROI 框，默认 true')
    parser.add_argument('--output-video-style', choices=['annotated', 'skeleton'], default='annotated', help='输出视频样式：原视频标注或仅显示球场、骨架和羽毛球')
    parser.add_argument('--generate-annotated-video', choices=['true', 'false'], default='false', help='是否生成标注视频（绘制与首次 H.264 编码），默认 false')
    parser.add_argument('--browser-video-reencode', choices=['true', 'false'], default='false', help='是否额外进行浏览器兼容重编码；仅生成标注视频时生效，默认 false')
    parser.add_argument('--display', choices=['true', 'false'], default='false', help='是否显示 OpenCV 预览窗口；开启会保留绘制但不导出视频，默认 false')
    parser.add_argument('--skeletons', choices=['true', 'false'], default='true', help='是否显示人体骨架，默认 true')
    parser.add_argument('--player-trajectories', choices=['true', 'false'], default='true', help='是否显示球员轨迹，默认 true')
    parser.add_argument('--court-trajectory', choices=['true', 'false'], default='true', help='是否显示球场轨迹，默认 true')
    parser.add_argument('--shuttlecock-trajectory', choices=['true', 'false'], default='true', help='是否显示羽毛球轨迹，默认 true')
    parser.add_argument('--player-stats', choices=['true', 'false'], default='true', help='是否显示球员统计信息，默认 true')
    parser.add_argument('--save-images', action='store_true', default=False, help='保存处理后的图像')
    parser.add_argument('--performance-stats', action='store_true', default=True, help='显示性能统计信息')
    parser.add_argument('--visualize-positions', choices=['true', 'false'], default='true', help='是否生成球员位置热力图和散点图，默认 true')
    parser.add_argument('--audio', choices=['true', 'false'], default='true', help='是否保留原视频音频，默认 true')
    parser.add_argument('--language', default='zh', choices=['zh', 'en'], help='选择界面语言 (zh/en)')
    args = parser.parse_args()

    load_runtime_dependencies()

    if args.language == 'en':
        from badminton_analysis.visualization.player_positions_en import analyze_player_positions
    else:
        from badminton_analysis.visualization.player_positions_zh import analyze_player_positions

    system = BadmintonAnalysisSystem(
        args.video_path,
        show_display=args.display == 'true',
        show_skeletons=args.skeletons == 'true',
        show_player_trajectories=args.player_trajectories == 'true',
        show_court_trajectory=args.court_trajectory == 'true',
        show_shuttlecock_trajectory=args.shuttlecock_trajectory == 'true',
        show_player_stats=args.player_stats == 'true',
        show_performance_stats=args.performance_stats,
        save_images=args.save_images,
        language=args.language,
        output_dir=args.output_dir,
        ball_model_path=args.ball_model,
        template_path=args.template_path,
        pose_mode=args.pose_mode,
        pose_family=args.pose_family,
        yolo_pose_model=args.yolo_pose_model,
        show_pose_roi=args.pose_roi == 'true',
        output_video_style=args.output_video_style,
        generate_annotated_video=args.generate_annotated_video == 'true',
        browser_video_reencode=args.browser_video_reencode == 'true',
        pose_imgsz=args.pose_imgsz,
        pose_sample_hz=args.pose_sample_hz,
        pose_conf=args.pose_conf,
        far_player_enhancement=args.far_player_enhancement == 'true',
        far_pose_roi=args.far_pose_roi,
        net_image_line=args.net_image_line,
    )

    system.keep_audio = args.audio == 'true'
    system.process_video()

    if args.visualize_positions == 'true':
        print("\n开始生成球员位置可视化...")
        analyze_player_positions(system.detections_path, os.path.join(system.save_dir, 'position_visualizations'), fps=system.fps)
        print("球员位置可视化完成")

if __name__ == "__main__":
    main()
