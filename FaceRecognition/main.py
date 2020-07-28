from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import detect_and_align
import argparse
import time
import cv2
import os
from keras.utils.data_utils import get_file
from wide_resnet import WideResNet
import dataSetGenerator
#tf.device('/cpu:0')
#from numba import vectorize
"""NUM_PARALLEL_EXEC_UNITS=6
from keras import backend as K

#import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })

session = tf.Session(config=config)

K.set_session(session)

os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"

os.environ["KMP_BLOCKTIME"] = "30"

os.environ["KMP_SETTINGS"] = "1"

os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"""
#@vectorize(['float32(float32, float32)'], target='cuda')













class IdData:
    """Keeps track of known identities and calculates id matches"""
    #initialize and establish tenserflow session
    def __init__(
        self, id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, distance_treshold
    ):
        print("Loading known identities: ", end="")
        self.distance_treshold = distance_treshold
        self.id_folder = id_folder
        self.mtcnn = mtcnn
        self.id_names = []
        
        image_paths = []
        ids = os.listdir(os.path.expanduser(id_folder))
        for id_name in ids:
            id_dir = os.path.join(id_folder, id_name)
            image_paths = image_paths + [os.path.join(id_dir, img) for img in os.listdir(id_dir)]

        print("Found %d images in id folder" % len(image_paths))
        aligned_images, id_image_paths = self.detect_id_faces(image_paths)
        feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
        self.embeddings = sess.run(embeddings, feed_dict=feed_dict)

        if len(id_image_paths) < 5:
            self.print_distance_table(id_image_paths)

    def detect_id_faces(self, image_paths):
        aligned_images = []
        id_image_paths = []
        for image_path in image_paths:
            image = cv2.imread(os.path.expanduser(image_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_patches, _, _,_ = detect_and_align.detect_faces(image, self.mtcnn)
            if len(face_patches) > 1:
                print(
                    "Warning: Found multiple faces in id image: %s" % image_path
                    + "\nMake sure to only have one face in the id images. "
                    + "If that's the case then it's a false positive detection and"
                    + " you can solve it by increasing the thresolds of the cascade network"
                )
            aligned_images = aligned_images + face_patches
            id_image_paths += [image_path] * len(face_patches)
            path = os.path.dirname(image_path)
            self.id_names += [os.path.basename(path)] * len(face_patches)

        return np.stack(aligned_images), id_image_paths

    def print_distance_table(self, id_image_paths):
        """Prints distances between id embeddings"""
        distance_matrix = pairwise_distances(self.embeddings, self.embeddings)
        image_names = [path.split("/")[-1] for path in id_image_paths]
        print("Distance matrix:\n{:20}".format(""), end="")
        [print("{:20}".format(name), end="") for name in image_names]
        for path, distance_row in zip(image_names, distance_matrix):
            print("\n{:20}".format(path), end="")
            for distance in distance_row:
                print("{:20}".format("%0.3f" % distance), end="")
        print()

    def find_matching_ids(self, embs):
        matching_ids = []
        matching_distances = []
        distance_matrix = pairwise_distances(embs, self.embeddings)
        for distance_row in distance_matrix:
            min_index = np.argmin(distance_row)
            if distance_row[min_index] < self.distance_treshold:
                matching_ids.append(self.id_names[min_index])
                matching_distances.append(distance_row[min_index])
            else:
                matching_ids.append(None)
                matching_distances.append(None)
        return matching_ids, matching_distances
    
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def load_model(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print("Loading model filename: %s" % model_exp)
        with gfile.FastGFile(model_exp, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    else:
        raise ValueError("Specify model file, not directory!")

def crop_face(imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)
    #this section is used to train the model with given images
def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            #CASE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"
            count=0
            # Setup models
            mtcnn = detect_and_align.create_mtcnn(sess, None)
            #count=0
            load_model(args.model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            # Load anchor IDs
            id_data = IdData(
                args.id_folder[0],
                mtcnn,
                sess,
                embeddings,
                images_placeholder,
                phase_train_placeholder,
                args.threshold,
            )
            test(mtcnn,id_data,args,sess,
                embeddings,
                images_placeholder,
                phase_train_placeholder,count)
         
    #this section isused to run the detecting and recognizing with age and gender prediction
def test(mtcnn,id_data,args,sess,
                embeddings,
                images_placeholder,
                phase_train_placeholder,count):          
            WRN_WEIGHTS_PATH = ".\\pretrained_models\\weights.18-4.06.hdf5"
            face_size = 64
            model = WideResNet(64, depth=16, k=8)()
            model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
            fpath = get_file('weights.18-4.06.hdf5',
            WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
            model.load_weights(fpath)
            cap = cv2.VideoCapture(0)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            #count=0
            show_landmarks = False
            show_bb = False
            show_id = True
            show_fps = False
            #show_train = False
            #real=0
            while True:
                start = time.time()
                _, frame = cap.read()

                #Locate faces and landmarks in frame
                face_patches, padded_bounding_boxes, landmarks,bounding_boxes = detect_and_align.detect_faces(frame, mtcnn)
                face_imgs = np.empty((len(face_patches),face_size,face_size, 3))
                for i, bb in enumerate(padded_bounding_boxes):
                    face_img, cropped = crop_face(frame,(bb[0],bb[1],180,180), margin=40, size=face_size)
                    (x, y, w, h) = cropped
                    #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                    face_imgs[i,:,:,:] = face_img
                if len(face_imgs) > 0:
                    results = model.predict(face_imgs)
                    predicted_genders = results[0]
                    ages = np.arange(0, 101).reshape(101, 1)
                    predicted_ages = results[1].dot(ages).flatten()
                    for i, face in enumerate(padded_bounding_boxes):
                      label = "{}, {}".format(int(predicted_ages[i]),
                                            "F" if predicted_genders[i][0] > 0.5 else "M")
                      
                       
                      font=cv2.FONT_HERSHEY_SIMPLEX
                      font_scale=1
                      thickness=2
                      
                      
                      
                      
                      
                      size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                      x, y = (face[0], face[1])
                      cv2.rectangle(frame, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
                      cv2.putText(frame, label, (face[0], face[1]), font, font_scale, (255, 255, 255), thickness)
                      #IdData.draw_label(frame, (face[0], face[1]), label)
                    face_patches = np.stack(face_patches)
                    feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
                    embs = sess.run(embeddings, feed_dict=feed_dict)

                    print("Matches in frame:")
                    matching_ids, matching_distances = id_data.find_matching_ids(embs)

                    for bb, landmark, matching_id, dist in zip(
                        padded_bounding_boxes, landmarks, matching_ids, matching_distances
                    ):
                        if matching_id is None:
                            matching_id = "Unknown"
                            print("Unknown! Couldn't fint match.")
                            
                            
                            
                            
                            
                            
                             
                       
                        
                        
                        
                        
                        
                        
                        
                        else:
                            print("Hi %s! Distance: %1.4f" % (matching_id, dist))

                        if show_id:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            #count +=1
                            cv2.putText(frame, matching_id, (bb[0], bb[3]), font, 1, (0, 225, 0), 1, cv2.LINE_AA)
                            #cv2.putText(frame, real , (100,100), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                            
                            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
                        if show_landmarks:
                            for j in range(5):
                                size = 1
                                top_left = (int(landmark[j]) - size, int(landmark[j + 5]) - size)
                                bottom_right = (int(landmark[j]) + size, int(landmark[j + 5]) + size)
                                cv2.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)
                        #if show_train:
                            
                            
                            
                else:
                    print("Couldn't find a face")

                end = time.time()

                seconds = end - start
                fps = round(1 / seconds, 2)

                if show_fps:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(fps), (0, int(frame_height) - 5), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow("frame", frame)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                elif key == ord("l"):
                    show_landmarks = not show_landmarks
                elif key == ord("b"):
                    show_bb = not show_bb
                elif key == ord("i"):
                    show_id = not show_id
                elif key == ord("f"):
                    show_fps = not show_fps
                elif key == ord("s"):
                    count=count+1
                    if count>10:
                        continue
                    gin=dataSetGenerator.gv()
                    key=input('Loaded images,press any key to continue')
                    if key == ord('y'):
                        
                       test(mtcnn,id_data,args,sess,
                embeddings,
                images_placeholder,
                phase_train_placeholder,count)
                    else:
                        test(mtcnn,id_data,args,sess,
                embeddings,
                images_placeholder,
                phase_train_placeholder,count)
                elif key == ord('t'):
                    cap.release()
                    cv2.destroyAllWindows()
                    main(args)
            






            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Path to model protobuf (.pb) file")
    parser.add_argument("id_folder", type=str, nargs="+", help="Folder containing ID folders")
    parser.add_argument("-t", "--threshold", type=float, help="Distance threshold defining an id match", default=0.75) 
    main(parser.parse_args())
