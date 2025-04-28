import timeit
import cv2
import numpy as np
import supervision as sv
from trackers import (
    PlayerTracker, 
    BallTracker, 
    KeypointsTracker, 
    PlayerKeypointsTracker,
    TrackingRunner,
)
from config import *


if __name__ == "__main__":
    
    t1 = timeit.default_timer()

    video_info = sv.VideoInfo.from_video_path(video_path=INPUT_VIDEO_PATH)
    fps, w, h, total_frames = (
        video_info.fps, 
        video_info.width,
        video_info.height,
        video_info.total_frames,
    )

    # Instantiate trackers
    keypoints_tracker = KeypointsTracker(
        model_path=KEYPOINTS_TRACKER_MODEL,
        batch_size=KEYPOINTS_TRACKER_BATCH_SIZE,
        model_type=KEYPOINTS_TRACKER_MODEL_TYPE,
        load_path=KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=KEYPOINTS_TRACKER_SAVE_PATH,
    )

    players_tracker = PlayerTracker(
        PLAYERS_TRACKER_MODEL,
        polygon_zone=None,
        batch_size=PLAYERS_TRACKER_BATCH_SIZE,
        annotator=PLAYERS_TRACKER_ANNOTATOR,
        show_confidence=True,
        load_path=PLAYERS_TRACKER_LOAD_PATH,
        save_path=PLAYERS_TRACKER_SAVE_PATH,
    )

    player_keypoints_tracker = PlayerKeypointsTracker(
        PLAYERS_KEYPOINTS_TRACKER_MODEL,
        train_image_size=PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE,
        batch_size=PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE,
        load_path=PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH,
    )

    ball_tracker = BallTracker(
        BALL_TRACKER_MODEL,
        BALL_TRACKER_INPAINT_MODEL,
        batch_size=BALL_TRACKER_BATCH_SIZE,
        median_max_sample_num=BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM,
        median=None,
        load_path=BALL_TRACKER_LOAD_PATH,
        save_path=BALL_TRACKER_SAVE_PATH,
    )

    runner = TrackingRunner(
        trackers=[
            players_tracker, 
            player_keypoints_tracker, 
            ball_tracker,
            keypoints_tracker,
        ],
        video_path=INPUT_VIDEO_PATH,
        inference_path=OUTPUT_VIDEO_PATH,
        start=0,
        end=MAX_FRAMES,
        collect_data=COLLECT_DATA,
    )

    runner.run()

    runner.export_ball_data(COLLECT_BALL_DATA_PATH)

    if COLLECT_DATA:
        data = runner.data_analytics.into_dataframe(runner.video_info.fps)
        data.to_csv(COLLECT_DATA_PATH)

    t2 = timeit.default_timer()
    print("Duration (min): ", (t2 - t1) / 60)
