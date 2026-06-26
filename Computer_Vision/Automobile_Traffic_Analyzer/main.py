from ultralytics import YOLOWorld, YOLO
import cv2
import easyocr
import torch
import os
import argparse
import global_vars as gv

def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "-url", "--video_stream_url",
        help = "URL address of video stream via RTSP."
    )
    received_args = args_parser.parse_args()
    if received_args.video_stream_url:
        video_stream = cv2.VideoCapture(received_args.video_stream_url)

    if not video_stream.isOpened():
        raise Exception("Error! Video stream was not found! Check your RTSP URL or camera device.")
    
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    car_detection_model = get_initialized_car_detection_model().to(compute_device)
    license_plate_detection_model = get_initialized_license_plate_detection_model().to(
        compute_device
    )
    ocr_reader = easyocr.Reader(['en'])

    window_name = gv.WINDOW_NAME
    window_width = gv.WINDOW_WIDTH_IN_PIXELS
    detect_automobiles_cache = {}
    
    while video_stream.isOpened():
        success, frame = video_stream.read()
        if success:
            cars_results = car_detection_model.track(frame, persist = True, verbose = False)
            if cars_results[0].boxes.id != None:
                update_detect_automobiles_cache_with_frame_cars(
                    license_plate_detection_model,
                    frame, detect_automobiles_cache,
                    cars_results,
                    ocr_reader
                )
                annotated_frame = get_annotated_frame_with_bounding_boxes(
                    frame, detect_automobiles_cache
                )
                show_frame_in_custom_window(annotated_frame, window_name, window_width)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("Video-stream was terminated or finished!")
            break
    video_stream.release()
    cv2.destroyAllWindows()

def get_initialized_car_detection_model():
    model = YOLOWorld(gv.AUTOMOBILE_DETECTION_MODEL_NAME)
    model.set_classes(gv.AUROMOBILE_DETECTION_CLASSES)
    return model

def get_initialized_license_plate_detection_model():
    if gv.LICENSE_PLATE_DETECTION_MODEL_NAME not in os.listdir():
        if os.name == 'posix':
            os.system(
                f"wget -O {gv.LICENSE_PLATE_DETECTION_MODEL_NAME} {gv.LICENSE_PLATE_DETECTION_MODEL_URL}"
            )
        else:
            os.system(
                f"curl -o {gv.LICENSE_PLATE_DETECTION_MODEL_NAME} -L {gv.LICENSE_PLATE_DETECTION_MODEL_URL}"
            )
    model_obj = YOLO(gv.LICENSE_PLATE_DETECTION_MODEL_NAME)
    return model_obj

def show_frame_in_custom_window(frame, name: str, width: int):
    resized_frame = get_resized_image(frame, width)
    cv2.imshow(name, resized_frame)

def get_resized_image(image, new_width: int):
    original_image_width = image.shape[1]
    original_image_height = image.shape[0]
    original_aspect_ratio = round(original_image_height / original_image_width, 1)
    new_height = int(original_aspect_ratio * new_width)
    resized_image = cv2.resize(
        image, (new_width, new_height),
        interpolation = cv2.INTER_LINEAR
    )
    return resized_image

def update_detect_automobiles_cache_with_frame_cars(
license_plate_detection_model, frame, cache: dict, cars_results, ocr_reader):
    cache_cars_IDs = cache.keys()
    
    frame_cars_boxes_obj = get_boxes_object_from_frame(cars_results[0])
    frame_cars_IDs = get_IDs_from_frame_boxes_obj(frame_cars_boxes_obj)
    frame_cars_boxes_xy_coordinates = get_boxes_xy_coordinates(frame_cars_boxes_obj)

    for (frame_car_ID, box_xy_coordinates) in zip(frame_cars_IDs, frame_cars_boxes_xy_coordinates):
        if frame_car_ID not in cache_cars_IDs:
            frame_car_ID = int(frame_car_ID.item())
            car_color = get_car_RGB_color(frame, box_xy_coordinates)
            car_license_plate_text = get_car_license_plate_text(
                license_plate_detection_model, frame,
                box_xy_coordinates,
                ocr_reader
            )
            cache[frame_car_ID] = {
                "box_xy_coordinates": box_xy_coordinates,
                "bounding_box_RGB_color": car_color,
                "license_plate": car_license_plate_text
            }
    car_ID_for_deleting = get_car_ID_already_not_on_screen(cache, frame_cars_IDs)
    if car_ID_for_deleting != 'ok':
        del cache[car_ID_for_deleting]

def get_car_ID_already_not_on_screen(cache: dict, frame_cars_IDs):
    for cache_car_ID in cache.keys():
        if cache_car_ID not in frame_cars_IDs:
            return cache_car_ID
    return "ok"

def get_boxes_object_from_frame(yolo_results):
    return yolo_results.boxes

def get_IDs_from_frame_boxes_obj(boxes_object) -> torch.Tensor:
    return boxes_object.id

def get_boxes_xy_coordinates(boxes_object) -> torch.Tensor:
    return boxes_object.xyxy

def get_car_RGB_color(frame, box_xy_coordinates: torch.Tensor):
    x1, y1, x2, y2 = map(int, box_xy_coordinates)
    car_area_tensor = torch.tensor(frame[y1:y2, x1:x2])
    car_area_width = car_area_tensor.shape[1]
    car_area_height = car_area_tensor.shape[0]
    car_bgr_pixels_sum_tensor = torch.sum(car_area_tensor, dim = (0, 1))
    car_averate_bgr_pixels_tensor = car_bgr_pixels_sum_tensor / (car_area_width * car_area_height + 1)
    car_averate_bgr_pixels_tensor = torch.round(car_averate_bgr_pixels_tensor)
    red = car_averate_bgr_pixels_tensor[2].item()
    green = car_averate_bgr_pixels_tensor[1].item()
    blue = car_averate_bgr_pixels_tensor[0].item()
    return (blue, green, red)

def get_car_license_plate_text(model, frame, car_box_xy_coordinates: torch.Tensor, ocr_reader):
    x1, y1, x2, y2 = map(int, car_box_xy_coordinates)
    car_area = frame[y1:y2, x1:x2]
    license_plates_results = model.track(car_area, verbose = False)
    frame_license_plates_boxes_obj = get_boxes_object_from_frame(license_plates_results[0])
    frame_license_plates_IDs = get_IDs_from_frame_boxes_obj(frame_license_plates_boxes_obj)
    frame_license_plates_xy_coordinates = get_boxes_xy_coordinates(frame_license_plates_boxes_obj)

    if len(frame_license_plates_xy_coordinates) > 0: 
        x1, y1, x2, y2 = map(int, frame_license_plates_xy_coordinates[0])
        license_plate_area = frame[y1:y2, x1:x2]
        ocr_results = ocr_reader.readtext(license_plate_area)
        if len(ocr_results) > 0:
            license_plate_text = ocr_results[0][1]
            return license_plate_text
    return ""

def get_annotated_frame_with_bounding_boxes(frame, cache: dict):
    annotated_frame = frame
    for car_ID in cache.keys():
        box_coordinates = cache[car_ID]["box_xy_coordinates"]
        x1, y1, x2, y2 = map(int, box_coordinates)
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            cache[car_ID]["bounding_box_RGB_color"],
            2
        )
        car_license_plate = cache[car_ID]["license_plate"]
        cv2.putText(
            annotated_frame, f"ID:{car_ID}:{car_license_plate}",
            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
            cache[car_ID]["bounding_box_RGB_color"], 3
        )
    return annotated_frame

if __name__ == "__main__":
    main()