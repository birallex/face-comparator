from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from PIL import Image

class Detector:
    def __init__(self, confidence_level = 0.4, default_image_resolution = (224, 224), detail_mode = False):
        self.confidence = confidence_level
        self.detail_mode = detail_mode
        self.default_size = default_image_resolution
        self.face_detector = MTCNN()
        self.score_detector = VGGFace(model='resnet50',
                                        include_top=False,
                                        input_shape=(224, 224, 3),
                                        pooling='avg'
                                        )


    def extract_faces_from_image(self, image_path):
        image = plt.imread(image_path)
        faces = self.face_detector.detect_faces(image)
        found_faces = []
        for face in faces:
            x1, y1, width, height = face['box']
            x2, y2 = x1 + width, y1 + height
            face_boundary = image[y1:y2, x1:x2]
            face_image = Image.fromarray(face_boundary)
            face_image = face_image.resize(self.default_size)
            face_array = asarray(face_image)
            found_faces.append(face_array)
            if self.detail_mode:
                plt.imshow(face_array)
                plt.show()
        return found_faces


    def extract_faces_from_stream(self, image_from_stream):
        # TODO
        pass

    def get_faces_score(self, faces):
        arrays = asarray(faces, 'float32')
        samples = preprocess_input(arrays, version=2)
        return self.score_detector.predict(samples)

    
    def compare_scores(self, scores):
        absolute_score = cosine(scores[0], scores[1])
        print(absolute_score)
        if  absolute_score < self.confidence:
            return True
        return False


if __name__ == "__main__":
    pass