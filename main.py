from utils import read_video, write_video


def main():
    print('main is running...')

    #read video
    video_frames = read_video('input_video/test_video.mp4')

    #write video
    save_video = write_video(video_frames, 'output_video/output_test_video.avi')

if __name__  == '__main__':
    main()