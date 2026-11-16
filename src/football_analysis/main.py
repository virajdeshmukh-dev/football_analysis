from football_analysis.utils import read_video, write_video
from football_analysis.football_trackers import Tracker

def main():
    # Run the complete video-analysis pipeline in its processing order.
    print('[main] Starting football-analysis pipeline')

    #read video
    video_frames = read_video('src/football_analysis/input_video/test_video.mp4')

    print(f'[main] Video read complete: {len(video_frames)} frame(s)')

    #tracker
    tracker = Tracker('src/football_analysis/models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='src/football_analysis/stubs/track_stubs.pkl')

    print('[main] Track result ready; starting annotation')

    #draw output
    #draw opuput video trackers
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    print(f'[main] Annotation complete: {len(output_video_frames)} frame(s)')

    #write video
    # Save the annotated result to the output-video directory.
    write_video(output_video_frames, 'src/football_analysis/output_video/output_test_video.avi')

    print('[main] Pipeline completed successfully')

if __name__  == '__main__':
    main()
