from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)
  
  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    pose_landmarks = pose_landmarks[11:]
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])

    for landmark in pose_landmarks_proto.landmark:
      x = int(landmark.x * annotated_image.shape[1])
      y = int(landmark.y * annotated_image.shape[0])
      cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

    # # Draw the pose landmarks.
    # pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    # pose_landmarks_proto.landmark.extend([
    #   landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    # ])
    # solutions.drawing_utils.draw_landmarks(
    #   annotated_image,
    #   pose_landmarks_proto,
    #   solutions.pose.POSE_CONNECTIONS,
    #   solutions.drawing_styles.get_default_pose_landmarks_style())

  return annotated_image

def apply_edge_detection(rgb_image):
  gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
  edges = cv2.Canny(gray_image, 85, 200)
  return edges

def find_closest_edge(edge_image, landmark_x, landmark_y, direction, from_exterior=True):
  height, width = edge_image.shape
  
  # Convert landmark coordinates to integers
  x = int(landmark_x)
  y = int(landmark_y)
  
  # Define direction vectors
  directions = {
    'up': (0, -1),
    'down': (0, 1),
    'left': (-1, 0),
    'right': (1, 0)
  }
  
  dx, dy = directions.get(direction, (0, 0))

  if(from_exterior):
    # Start from the image boundary based on direction
    if direction == 'left':
      x = width - 1  # Start from right edge
    elif direction == 'right':
      x = 0  # Start from left edge
    elif direction == 'up':
      y = height - 1  # Start from bottom edge
    elif direction == 'down':
      y = 0  # Start from top edge

    # Store target landmark position to know when to stop
    target_x = int(landmark_x)
    target_y = int(landmark_y)

    while 0 <= x < width and 0 <= y < height:
      if edge_image[y, x] == 255:  # Found white pixel (edge)
        return (x, y)
      # Stop if we've passed the landmark position
      if (direction in ['left', 'right'] and x == target_x) or \
        (direction in ['up', 'down'] and y == target_y):
        break
      x += dx
      y += dy

  else :
    while 0 <= x < width and 0 <= y < height:
      if edge_image[y, x] == 255:  # Found white pixel (edge)
        return (x, y)
      x += dx
      y += dy
  
  return None  # No edge found in that direction

def get_middle_point(landmark1, landmark2, edges):
  x1 = int(landmark1.x * edges.shape[1])
  y1 = int(landmark1.y * edges.shape[0])
  x2 = int(landmark2.x * edges.shape[1])
  y2 = int(landmark2.y * edges.shape[0])
  
  middle_x = (x1 + x2) // 2
  middle_y = (y1 + y2) // 2
  
  return (middle_x, middle_y)

def get_point_at_ratio(point1, point2, ratio):
  x1, y1 = point1
  x2, y2 = point2

  x = int(x1 + ratio * (x2 - x1))
  y = int(y1 + ratio * (y2 - y1))

  return (x, y)

def get_distance(point1, point2):
  x1, y1 = point1
  x2, y2 = point2

  return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_measurements(img_path, detector, height=None, side_ratio=None) :
  # Front view measurements
  image = mp.Image.create_from_file(img_path)
  detection_result = detector.detect(image)
  annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
  # cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
  # cv2.imshow("Pose Detection", annotated_image)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

  edges = apply_edge_detection(image.numpy_view())
  # cv2.namedWindow('Edge Detection', cv2.WINDOW_NORMAL)
  # cv2.imshow("Edge Detection", edges)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

  hips_span, waist_span, bust_span, biceps_span, arm_length, arm_span = None, None, None, None, None, None

  pose_landmarks_list = detection_result.pose_landmarks
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])

    # Hips meaurements
    landmark = pose_landmarks_proto.landmark[24]
    x = int(landmark.x * edges.shape[1])
    y = int(landmark.y * edges.shape[0])
    point1 = find_closest_edge(edges, x, y, 'right')
    cv2.circle(annotated_image, point1, 5, (255, 0, 0), -1)

    landmark = pose_landmarks_proto.landmark[23]
    x = int(landmark.x * edges.shape[1])
    y = int(landmark.y * edges.shape[0])
    point2 = find_closest_edge(edges, x, y, 'left')
    cv2.circle(annotated_image, point2, 5, (0, 0, 255), -1)

    hips_span = get_distance(point1, point2)

    # Shoulders middle point & measurements
    shoulders_middle = get_middle_point(pose_landmarks_proto.landmark[11], pose_landmarks_proto.landmark[12], edges)
    cv2.circle(annotated_image, shoulders_middle, 5, (255, 255, 0), -1)

    # Shoulders middle point
    hips_middle = get_middle_point(pose_landmarks_proto.landmark[24], pose_landmarks_proto.landmark[23], edges)
    cv2.circle(annotated_image, hips_middle, 5, (255, 255, 0), -1)

    # Waist measurements
    waist_point = get_point_at_ratio(hips_middle, shoulders_middle, 2/5)
    cv2.circle(annotated_image, waist_point, 5, (255, 0, 255), -1)
    point1 = find_closest_edge(edges, waist_point[0], waist_point[1], 'right')
    cv2.circle(annotated_image, point1, 5, (255, 0, 0), -1)
    point2 = find_closest_edge(edges, waist_point[0], waist_point[1], 'left')
    cv2.circle(annotated_image, point2, 5, (0, 0, 255), -1)

    waist_span = get_distance(point1, point2)

    # Bust measurements
    bust_point = get_point_at_ratio(hips_middle, shoulders_middle, 7/10)
    cv2.circle(annotated_image, bust_point, 5, (255, 0, 255), -1)
    point1 = find_closest_edge(edges, bust_point[0], bust_point[1], 'right', from_exterior=False)
    cv2.circle(annotated_image, point1, 5, (255, 0, 0), -1)
    point2 = find_closest_edge(edges, bust_point[0], bust_point[1], 'left', from_exterior=False)
    cv2.circle(annotated_image, point2, 5, (0, 0, 255), -1)

    bust_span = get_distance(point1, point2)

    # Arm measurements
    arm_point = get_middle_point(pose_landmarks_proto.landmark[11], pose_landmarks_proto.landmark[13], edges)
    cv2.circle(annotated_image, arm_point, 5, (255, 0, 255), -1)
    point1 = find_closest_edge(edges, arm_point[0], arm_point[1], 'down', from_exterior=False)
    cv2.circle(annotated_image, point1, 5, (255, 0, 0), -1)
    point2 = find_closest_edge(edges, arm_point[0], arm_point[1], 'down')
    cv2.circle(annotated_image, point2, 5, (0, 0, 255), -1)

    biceps_span = get_distance(point1, point2)

    landmark15 = pose_landmarks_proto.landmark[15]
    landmark16 = pose_landmarks_proto.landmark[16]

    x1 = landmark15.x * edges.shape[1]
    y1 = landmark15.y * edges.shape[0]
    x2 = landmark16.x * edges.shape[1]
    y2 = landmark16.y * edges.shape[0]

    arm_span = get_distance((x1, y1), (x2, y2))

    landmark11 = pose_landmarks_proto.landmark[11]
    landmark13 = pose_landmarks_proto.landmark[13]

    x1 = landmark11.x * edges.shape[1]
    y1 = landmark11.y * edges.shape[0]
    x2 = landmark13.x * edges.shape[1]
    y2 = landmark13.y * edges.shape[0]

    arm_length = get_distance((x1, y1), (x2, y2))

  # cv2.namedWindow('New Landmarks', cv2.WINDOW_NORMAL)
  # cv2.imshow("New Landmarks", annotated_image)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  
  return hips_span, waist_span, bust_span, biceps_span, arm_length, arm_span

def ellipse_circumference(a, b):
  # Using Ramanujan's approximation
  h = ((a - b) / (a + b))**2
  return math.pi * (a + b) * (1 + (3*h)/(10 + math.sqrt(4 - 3*h)))

if __name__ == '__main__':
  base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
  options = vision.PoseLandmarkerOptions(
      base_options=base_options,
      output_segmentation_masks=True)
  detector = vision.PoseLandmarker.create_from_options(options)
  hips_span_front, waist_span_front, bust_span_front, biceps_span_front, arm_length_front, arm_span = get_measurements('front.jpg', detector)
  hips_span_side, waist_span_side, bust_span_side, biceps_span_side, arm_length_side, _ = get_measurements('side.jpg', detector)

  height = 185

  front_ratio = height * 0.9189 / arm_span
  hips_a = hips_span_front * front_ratio
  waist_a = waist_span_front * front_ratio
  bust_a = bust_span_front * front_ratio
  biceps_a = biceps_span_front * front_ratio

  side_ratio = front_ratio * arm_length_front / arm_length_side
  hips_b = hips_span_side * side_ratio
  waist_b = waist_span_side * side_ratio
  bust_b = bust_span_side * side_ratio
  biceps_b = biceps_span_side * side_ratio

  hips_circumference = ellipse_circumference(hips_a/2, hips_b/2)
  waist_circumference = ellipse_circumference(waist_a/2, waist_b/2)
  bust_circumference = ellipse_circumference(bust_a/2, bust_b/2)
  biceps_circumference = ellipse_circumference(biceps_a/2, biceps_b/2)

  print(f"Hips circumference: {hips_circumference:.2f} cm")
  print(f"Waist circumference: {waist_circumference:.2f} cm")
  print(f"Bust circumference: {bust_circumference:.2f} cm")
  print(f"Biceps circumference: {biceps_circumference:.2f} cm")