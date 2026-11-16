import cv2


#a function to read the video
def read_video(video_path: str):
    # Open the input video before reading its frames.
    print(f"[read_video] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Release the capture and report the result before returning it.
    cap.release()
    print(f"[read_video] Finished reading {len(frames)} frame(s)")
    return frames

#a function to write the video
def write_video(output_video_frames, output_video_path):
    # Report the intended output before creating the video writer.
    print(
        f"[write_video] Writing {len(output_video_frames)} frame(s) to "
        f"{output_video_path}"
    )
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    # Confirm that the writer has been released after saving all frames.
    print(f"[write_video] Finished writing video: {output_video_path}")
