import cv2, dlib, sys
import numpy as np

scaler = 0.3

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture('samples/girl.mp4')

while True:
  ret, img = cap.read()
  if not ret:
    break

  img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))

  faces = detector(img, 1)

  if len(faces) == 0:
    print('no faces!')

  shapes2D = []
  for face in faces:
    dlib_shape = predictor(img, face)
    
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    for s in shape_2d:
      cv2.circle(img, tuple(s), 1, (255, 255, 255), 2, cv2.LINE_AA)

  cv2.imshow('img', img)
  if cv2.waitKey(1) == ord('q'):
    sys.exit(1)