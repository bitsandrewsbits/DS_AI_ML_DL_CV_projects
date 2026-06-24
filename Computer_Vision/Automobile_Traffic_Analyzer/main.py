from ultralytics import YOLOWorld
import cv2
import global_vars as gv

def main(rtsp_URL: str):
    video_stream = cv2.VideoCapture(rtsp_URL)
    if not video_stream.isOpened():
        raise Exception("Error! Video stream was not found!")
    
    model = get_initialized_detection_model()
    window_name = "YOLO_v8_World Tracking"
    window_width = 1280
    
    while video_stream.isOpened():
        success, frame = video_stream.read()
        if success:
            results = model.track(frame, persist = True)
            print(results[0].orig_img.shape)
            annotated_frame = results[0].plot()
            show_frame_in_custom_window(annotated_frame, window_name, window_width)
            # exit by 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    video_stream.release()
    cv2.destroyAllWindows()

def get_initialized_detection_model():
    model = YOLOWorld(gv.AUTOMOBILE_DETECTION_MODEL_NAME)
    model.set_classes(gv.DETECTION_CLASSES)
    return model

def show_frame_in_custom_window(frame, name: str, width: int):
    cv2.namedWindow(name)
    resized_frame = get_resized_image(frame, width)
    cv2.imshow(name, resized_frame)

def get_resized_image(image, new_width: int):
    original_image_width = image.shape[1]
    original_image_height = image.shape[0]
    original_aspect_ratio = round(original_image_height / original_image_width, 1)
    new_height = int(original_aspect_ratio * new_width)
    # print("new width, height:", new_width, new_height)
    resized_image = cv2.resize(
        image, (new_width, new_height),
        interpolation = cv2.INTER_LINEAR
    )
    return resized_image

if __name__ == "__main__":
    video_stream_URL = "test_automob_traffic_video_2.mp4"
    main(video_stream_URL)