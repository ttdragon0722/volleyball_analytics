"""
Demo script for volleyball analytics ML Manager.

This script provides two main functions:
1. run_object_detection: Detects and tracks ball with trailing path, detects actions
2. run_video_classification: Classifies game state every 30 frames and visualizes results
"""
import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from rich.progress import Progress

from ml_manager.ml_manager import MLManager
from ml_manager.utils.tools import det2supervision, draw_trajectory

from vb_gui.calibration.database import CalibrationDB
from vb_gui.calibration.calibration_gui import create_gui

from court_mapping import MapperConfig, MapperState, map_frame, provider_from_calibration
from court_mapping.airborne import airborne_tracker_ids
from court_mapping.export import build_export, write_export
from court_mapping.minimap import MinimapStyle, overlay_minimap, render_minimap

# YOLO input size for player (pose) detection. 640 misses most far-court players in
# 1080p broadcast footage; 1280 recovers them at almost the same speed (~40 ms/frame).
PLAYER_DETECTION_IMGSZ = 1280


def run_object_detection(ml_manager: MLManager,
                         video_path: str,
                         output_path: str,
                         court_calibration_json: dict,
                         court_json_output: str | None = None) -> None:
    """
    Run object detection on video with ball tracking, action detection, and player detection.
    
    This function:
    - Loads the ML Manager
    - Detects and tracks ball with 8-frame trailing path
    - Detects actions (no tracking needed for actions)
    - Detects players and visualizes them with triangles
    - Visualizes ball with yellow circles, actions with class-specific colored boxes, and players with green triangles
    - Saves the output video
    
    Args:
        court_calibration_json:
        ml_manager: The MLManager instance.
        video_path: Path to input video file.
        output_path: Path to save output video with visualizations
    """
    print("Initializing ML Manager...")

    ml_manager.check_models()

    print("Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Get video properties
    width, height, fps, _, total_frames = [int(cap.get(prop)) for prop in range(3, 8)]
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

    # setup output folder
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize supervision annotators
    ball_annotator = sv.LabelAnnotator(color=sv.Color.YELLOW, text_thickness=1, text_scale=0.5)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    # Players annotators
    triangle_annotator = sv.TriangleAnnotator(color=sv.Color.GREEN)
    # ellipse_annotator = sv.EllipseAnnotator(color=sv.Color.GREEN)

    # Define colors for different action classes (class_id -> color)
    COLORS = {
        "ball": sv.Color.RED,
        "block": sv.Color.BLUE,
        "receive": sv.Color.from_hex('#800080'),
        "set": sv.Color.from_hex("#FFA500"),
        "spike": sv.Color.from_hex("#FFC0CB")
    }

    # Court mapping (bird's-eye projection) state
    court_provider = provider_from_calibration(court_calibration_json)
    court_homography = court_provider.homography
    print(f"Court calibration: mean reprojection error {court_homography.mean_error_px:.2f}px, "
          f"max {court_homography.max_error_px:.2f}px over {len(court_homography.inlier_names)} ground points")
    mapper_config = MapperConfig.for_fps(fps)
    mapper_state = MapperState()
    minimap_style = MinimapStyle(pixels_per_metre=max(10.0, height / 1080 * 20.0))
    court_frames = []

    # Tracking state
    tracker = sv.ByteTrack()
    ball_trajectory = []
    frame_count = 0
    print("Processing video frames...")

    with Progress() as progress:
        task = progress.add_task(
            "[green]Processing video...",
            total=total_frames
        )

        while frame_count < total_frames:
            ret, frame = cap.read()
            progress.update(task, advance=1)

            if not ret:
                progress.update(task, advance=30)
                break

            frame_count += 1
            # Detect actions and ball at default resolution
            actions = ml_manager.detect_actions(frame, conf_threshold=0.25, iou_threshold=0.45)
            ball = ml_manager.detect_ball(frame, conf_threshold=0.25, iou_threshold=0.45)
            # Players at a higher input size: far-court players are too small at the default 640
            players = ml_manager.player_detector.detect(
                frame, 0.25, 0.45, imgsz=PLAYER_DETECTION_IMGSZ)

            # Annotate frame
            annotated_frame = frame.copy()

            # Process ball detections
            if ball:
                ball_detections_sv = ball.to_supervision()
                annotated_frame = ball_annotator.annotate(
                    scene=annotated_frame,
                    detections=ball_detections_sv,
                    labels=["ball"]
                )
                # Update ball trajectory for tracking
                ball_trajectory.append(ball.bbox.center)

            # Process action detections
            action_detections_sv = det2supervision(detections=actions)

            # Process player detections
            player_detections_sv = det2supervision(detections=players, tracking_instance=tracker)

            # Players doing a jump action (block/spike/serve/set): freeze them at the take-off point
            action_names = [ml_manager.action_detector.action_id2name(c) for c in action_detections_sv.class_id] \
                if len(action_detections_sv) > 0 else []
            airborne_ids = airborne_tracker_ids(
                player_detections_sv.xyxy, player_detections_sv.tracker_id,
                action_detections_sv.xyxy, action_names,
                mapper_config.airborne.jump_actions, mapper_config.airborne.min_iou,
            )

            # Project players onto the court plane; drop people outside court (+1 m margin);
            # players missed for up to 1 s stay on the minimap (semi-transparent)
            mapper_state, court_record = map_frame(
                mapper_state, frame_count - 1,
                player_detections_sv.xyxy, player_detections_sv.tracker_id,
                court_provider, mapper_config, frame, airborne_ids=airborne_ids,
            )
            court_frames.append(court_record)
            if len(player_detections_sv) > 0 and player_detections_sv.tracker_id is not None:
                on_court_ids = [p.tracker_id for p in court_record.observed_players]
                player_detections_sv = player_detections_sv[np.isin(player_detections_sv.tracker_id, on_court_ids)]

            # Draw ball trajectory (8 frames trailing)
            if ball_trajectory:
                draw_trajectory(annotated_frame, ball_trajectory, 12)


            # Draw action detections with class-specific colors
            if len(action_detections_sv) > 0:
                # Group detections by class_id for color-specific annotation
                for class_id in np.unique(action_detections_sv.class_id):
                    # Filter detections for this class
                    class_mask = action_detections_sv.class_id == class_id
                    class_detections: sv.Detections = action_detections_sv[class_mask]

                    # Get color for this class, default to RED if not found
                    action = ml_manager.action_detector.yolo_module.id2class(class_id)
                    color = COLORS.get(action, sv.Color.WHITE)

                    # Create annotator for this specific color
                    class_box_annotator = sv.BoxAnnotator(thickness=2, color=color)

                    # Annotate with class-specific color
                    annotated_frame = class_box_annotator.annotate(
                        scene=annotated_frame,
                        detections=class_detections
                    )

                    # Get labels for this class
                    class_labels = [
                        ml_manager.action_detector.action_id2name(det)
                        for det in class_detections.class_id
                    ]
                    # class_labels = [action_labels[i] for i, mask in enumerate(class_mask) if mask]
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame,
                        detections=class_detections,
                        labels=class_labels
                    )

            # Draw player detections with triangles (green)
            if len(player_detections_sv) > 0:
                annotated_frame = triangle_annotator.annotate(
                    scene=annotated_frame,
                    detections=player_detections_sv
                )
                labels = [
                    f"Player #{tracker_id}"
                    for tracker_id in player_detections_sv.tracker_id
                ]
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=player_detections_sv,
                    labels=labels
                )

            # Bird's-eye minimap in the top-right corner
            annotated_frame = overlay_minimap(
                annotated_frame, render_minimap(court_record.players, minimap_style), minimap_style)

            # Write frame to output video
            out.write(annotated_frame)

        # Cleanup
        cap.release()
        out.release()

        court_json_path = court_json_output or Path(output_path).with_suffix(".court.json").as_posix()
        write_export(court_json_path, build_export(Path(video_path).name, fps, court_frames, court_homography))

        print(f"Object detection completed (ball, actions, players). Output saved to: {output_path}")
        print(f"Player court coordinates saved to: {court_json_path}")


def run_video_classification(ml_manager: MLManager, video_path: str, output_path: str) -> None:
    """
    Run game state classification on video every 30 frames.

    This function:
    - Loads the ML Manager
    - Classifies game state every 30 frames
    - Visualizes classification results on the video
    - Saves the output video

    Args:
        ml_manager: Model Manager that loads and manages models.
        video_path: Path to input video file
        output_path: Path to save output video with visualizations
    """
    print("Initializing ML Manager...")

    # Check if game state classification model is available
    if not ml_manager.is_model_available('game_state_classification'):
        raise RuntimeError("Game state classification model not available")

    print("Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Get video properties
    width, height, fps, _, total_frames = [int(cap.get(prop)) for prop in range(3, 8)]
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

    # setup output folder
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Classification parameters
    frame_buffer = []
    current_confidence = 0.0

    print("Processing video frames...")
    frame_count = 0

    with Progress() as progress:
        task = progress.add_task(
            "[green]Processing video...",
            total=total_frames
        )

        while True:
            ret, frame = cap.read()
            frame_count += 1
            progress.update(task, advance=1)

            if not ret:
                progress.update(task, advance=30)
                break

            # Add frame to buffer
            frame_buffer.append(frame.copy())

            if len(frame_buffer) != 30:
                continue

            # Classify game state every 30 frames
            game_state_result = ml_manager.classify_game_state(frame_buffer)

            current_game_state = game_state_result.predicted_class
            current_confidence = game_state_result.confidence

            progress.update(
                task,
                description=(
                    f"[green]Frame {frame_count} | "
                    f"State: {current_game_state} | "
                    f"Conf: {current_confidence:.3f}"
                )
            )

            # Annotate frame with game state
            annotated_frame = frame.copy()

            state_text = f"Game State: {current_game_state}"
            confidence_text = f"Confidence: {current_confidence:.3f}"
            frame_text = f"Frame: {frame_count}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            text_size = cv2.getTextSize(state_text, font, font_scale, thickness)[0]

            cv2.rectangle(annotated_frame, (10, 10), (text_size[0] + 20, 100), (0, 0, 0), -1)

            cv2.rectangle(annotated_frame, (10, 10), (text_size[0] + 20, 100), (255, 255, 255), 2)

            cv2.putText(annotated_frame, state_text, (15, 35), font, font_scale, (255, 255, 255), thickness)

            cv2.putText(annotated_frame, confidence_text, (15, 60), font, font_scale, (255, 255, 255), thickness)

            cv2.putText(annotated_frame, frame_text, (15, 85), font, font_scale, (255, 255, 255), thickness)

            if current_game_state == "Play":
                color = (0, 255, 0)
            elif current_game_state == "No Play":
                color = (0, 0, 255)
            elif current_game_state == "Service":
                color = (255, 0, 255)
            else:
                color = (255, 255, 255)

            cv2.rectangle(annotated_frame, (10, 10), (text_size[0] + 20, 100), color, 3)

            out.write(annotated_frame)

    # Cleanup
    cap.release()
    out.release()
    ml_manager.cleanup()

    print(f"Video classification completed. Output saved to: {output_path}")


def main():
    # TODO:
    #  Design the software based on this: https://chatgpt.com/share/6a6deeb9-ba74-83eb-8290-3a2afbf184f4
    #  Inference:
    #  1. Add supervision tracker to players.
    #  2. Add Heatmap for each player later.
    #  3. Check if there are better and easier to work with 'trackers'.
    #  4. Add court detection to the pipeline and filter the persons outside court.
    #  5. Add integration of video classification and object detection together and API endpoints in
    #     the inference.
    #  6. Update the GIFs and videos on the readme file with the new demos.
    #  7. After the video classification has created the slice, add it to a job queue and process the
    #     slices with object detection.
    #  8. 2D visualization of the court and the ball bounce points.
    #  Front-end:
    #  1. Page 1: Inference page -> Object detection and Video slicing
    #  2. Page 2: Annotation tool -> Being able to annotate and use the models to help out.
    #  3. Page 3: Set the court corners and the 1/3rd line. In order to omit the court detection lines.
    #  4. Page 4: Analysis page -> Shows the court on 2d (or maybe 3d) and plots ball, players locations on the court.
    """
    Example usage of the demo functions.
    """
    # Example video path (update with your actual video path)
    root_path = Path().cwd().parent
    db_dir = root_path.joinpath("db")
    db_dir.mkdir(parents=True, exist_ok=True)  # SQLite creates the file but not its parent folder
    db_path = db_dir.joinpath('calibration.db').as_posix()
    ml_manager = MLManager()
    # output_classification = "../output/IRN_vs_FRA_video_classification.mp4"


    # Calibration of the video for court points.
    db = CalibrationDB(db_path=db_path)

    # Select video using the GUI.
    result_json, video_path = create_gui(
        video_path=None,
        db_instance=db,
    )

    output_detection = root_path.joinpath("output").joinpath(video_path.stem + '_object_detection.mp4').as_posix()

    #
    # Create output directory if it doesn't exist
    Path("output").mkdir(exist_ok=True)

    print("Running object detection demo...")
    run_object_detection(ml_manager, video_path, output_detection, result_json)

    # print("\nRunning video classification demo...")
    # run_video_classification(ml_manager, video_path, output_classification)

    print("\nDemo completed successfully!")
    print(f"Object detection output: {output_detection}")
    # print(f"Video classification output: {output_classification}")


if __name__ == "__main__":
    main()
