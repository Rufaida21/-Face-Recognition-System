import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
dataset_path = "dataset"

def get_images_with_ids(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    faces = []
    ids = []
    for image_path in image_paths:
        face_img = Image.open(image_path).convert("L")  # Convert to grayscale
        face_np = np.array(face_img, np.uint8)
        id = int(os.path.split(image_path)[-1].split(".")[1])
        print(f"Training image for ID: {id}")
        faces.append(face_np)
        ids.append(id)
        cv2.imshow("Training", face_np)
        cv2.waitKey(10)
    return np.array(ids), faces

ids, faces = get_images_with_ids(dataset_path)
recognizer.train(faces, ids)
recognizer.save("Recognizer/trainingdata.yml")
cv2.destroyAllWindows()
print("Training completed and model saved to 'Recognizer/trainingdata.yml'")
