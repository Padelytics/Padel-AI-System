# Padel Analytics 🤖🎾

This repository applies computer vision techniques to extract valuable insights from a padel game recording, including:
- Position and velocity of each player.
- Position and velocity of the ball.
- 2D court projection.
- Heatmaps.
- Ball velocity associated with different strokes.
- Player error rate.

## Project Description

It is with great pleasure that I announce my new project **AI Powered Padel Analytics**.  
Merging my interest in **padel** and **computer vision**, I developed a first iteration that uses **object tracking** and **keypoint detection** for data analytics from a match video recording.

Several computer vision models were trained to:
1. Track the position of individual players;
2. Estimate players' pose with 13 keypoints;
3. Classify players' technical gestures (e.g., backhand volley, forehand volley, bandeja, topspin smash, etc);
4. Predict ball hits.

The goal of this project is to provide precise and robust analytics using only a padel game recording.  
This implementation can be used to:
- Upgrade live broadcasts by providing interesting data to the audience or storing it for future analysis;
- Generate valuable insights for padel coaches and players to enhance their continuous improvement journey.

## Models Used

- Players object tracking: YOLO + ByteTrack 🏃
- Players keypoints detection: YOLO-Pose 🏃
- Ball tracking: TrackNetv3 🥎
- Court keypoints detection: YOLO-Pose 🏟

Court keypoints are used to perform a **homography projection** into a **2D court representation** of players' feet and ball positions.

## Data Collected

- Players' position, velocity (Vx, Vy), and acceleration (Ax, Ay)
- Distance covered by each player
- Players' pose keypoints (13 degrees of freedom)
- Ball position, velocity, and acceleration

## Future Work

- Use players' pose keypoints detections to classify technical gestures more accurately;
- Predict when players hit the ball:  
  Unlike tennis, in padel we cannot rely solely on ball velocity vector direction changes due to wall bounces.

## Resources

- **Datasets**: [Download Here](https://drive.google.com/drive/folders/1gmmYLORGo_OKdc5_W_vq3L5U29TY0JjD?usp=drive_link)
- **Training**: [Download Here](https://drive.google.com/drive/folders/1XRiOEG7ok5TJ0-2knlAVg5I57Y5RZnTy?usp=drive_link)
- **Pre-trained Model Weights**: [Download Here](https://drive.google.com/drive/folders/1ylOTG9M81RdT9LbS51_TevcTOtIhKuS6?usp=sharing)

## Contact

Feel free to reach out to me if you are looking for a possible collaboration or if you have any questions.
