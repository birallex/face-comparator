from detector import Detector
from matplotlib import pyplot as plt

if __name__ == "__main__":
    comparator = Detector(confidence_level = 0.45, detail_mode = True)
    faces = [comparator.extract_faces_from_image(image_path)
                        for image_path in ['faces/1.jpg', 'faces/v1.jpg']]
                        #for image_path in ['faces/2.jpg', 'faces/3.jpg']]
    faces = [faces[0][0], faces[1][0]]
    scores = comparator.get_faces_score(faces)
    if comparator.compare_scores(scores):
        print("Faces Matched")
    else:
        print("Faces NOT Matched")
