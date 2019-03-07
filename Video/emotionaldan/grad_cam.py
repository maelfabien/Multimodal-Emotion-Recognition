from skimage.transform import resize
from skimage import color
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from utils import find_closest

emotionDict7 = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness',
                3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}

IMGSIZE = 224


def get_gradcam(batch_img, batch_label, modelPath, dan, tf_sess,
                img_mask=1, img_size=IMGSIZE, conv_layer='S2_Conv4a', logging=False):
    # TODO: fix for batch size > 1

    images = tf.placeholder(tf.float32, [None, img_size, img_size, 1])
    labels = tf.placeholder(tf.float32, [None, ])

    # gradient for partial linearization. We only care about target
    # visualization class.
    y_c = tf.reduce_sum(tf.multiply(dan['S2_Emotion'], labels), axis=1)

    # Get last convolutional layer gradient for generating gradCAM
    # visualization
    target_conv_layer = dan[conv_layer]
    target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]

    # Run inference of prediction class
    prob, softmax = tf_sess.run([dan['Pred_emotion'], dan['softmax']],
                       {dan['InputImage']: batch_img,
                        dan['S1_isTrain']: False,
                        dan['S2_isTrain']: False})
    if logging:
        # print(prob)
        # print(softmax)
        print('Predicted emotion:', emotionDict7[prob[0]])
        print('True emotion', emotionDict7[batch_label[0]])
        for i in range(len(softmax[0])):
            print(emotionDict7[i], softmax[0][i])
        

    # Do not generate cam if predicted class is not correct
    if prob[0] != batch_label[0]:
        return []

    target_conv_layer_value, target_conv_layer_grad_value = tf_sess.run(
        [target_conv_layer, target_conv_layer_grad],
        feed_dict={images: batch_img,
                   labels: batch_label,
                   dan['InputImage']: batch_img,
                   dan['S1_isTrain']: False,
                   dan['S2_isTrain']: False})

    batch_size = len(batch_img)

    for i in range(batch_size):

        output, grads_val = target_conv_layer_value[i], target_conv_layer_grad_value[i]
        weights = np.mean(grads_val, axis=(0, 1))

        cam = np.zeros(output.shape[0: 2], dtype=np.float32)
        for j, w in enumerate(weights):
            cam += w * output[:, :, j]

        # Passing through ReLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)  # scale 0 to 1.0
        cam = resize(cam, (img_size, img_size), preserve_range=True)

        return cam


def get_heatmap(cam):
    """Returns heatmap of generated gradcam"""
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    return cam_heatmap


def overlay_cam(img, cam, alpha, show=True, outFile=None):
    if len(cam) > 0:
        cam_heatmap = get_heatmap(cam)

        img = img.astype(float)
        img = img - np.min(img)
        img = img / np.max(img)

        # Construct RGB version of grey-level image
        img_color = np.dstack((img, img, img))

        # Convert the input image and color mask to Hue Saturation Value (HSV)
        # colorspace
        img_hsv = color.rgb2hsv(img_color)
        color_mask_hsv = color.rgb2hsv(cam_heatmap)

        # Replace the hue and saturation of the original image
        # with that of the color mask
        img_hsv[..., 0] = color_mask_hsv[..., 0]
        img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

        img_masked = color.hsv2rgb(img_hsv)

        if show:
            fig = plt.figure(figsize=(12, 16))
            ax = fig.add_subplot(222)
            plt.imshow(img_masked)
            ax.axis('off')
            ax.set_title('Grad-CAM')

            plt.show()

            if outFile:
                plt.savefig(outFile)

        return img_masked


def plot_mean_cam(cams):
    cams_num = [
        c for c in cams if (
            not np.isnan(c).any()) and (
            len(c) > 0)]
    print('Nb of images: {}, non nan: {}'.format(len(cams), len(cams_num)))
    if len(cams_num) > 0:
        cams_sum = sum(cams_num) / len(cams_num)
        cam_heatmap = cv2.applyColorMap(
            np.uint8(255 * cams_sum), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        plt.imshow(cam_heatmap)
        plt.show()


def get_mean_cam(cams):
    cams_num = [
        c for c in cams if (
            not np.isnan(c).any()) and (
            len(c) > 0)]
#     print('Nb of images: {}, non nan: {}'.format(len(cams), len(cams_num)))
    if len(cams_num) > 0:
        cams_sum = sum(cams_num) / len(cams_num)
        cam_heatmap = cv2.applyColorMap(
            np.uint8(255 * cams_sum), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        plt.imshow(cam_heatmap)
        return cams_sum


def find_closest_image_to_gradcam(emotion_id, gradcams, id_to_img_dict):
    mean_gradcam = get_mean_cam(gradcams[emotion_id - 1])
    res, res_i = find_closest(gradcams[emotion_id - 1], mean_gradcam)
    res_image = np.reshape(id_to_img_dict[emotion_id, res_i], (224, 224))
    return res_image, res


def visualize(batch_img, batch_label, modelPath, dan, sess, img_mask=1, batch_size=1,
              img_size=224, alpha=0.5, conv_layer='S2_Conv4b', vis_type='overlay', outFile=None):
    """Plot GradCam visualization for given image"""
    cam = get_gradcam(batch_img, batch_label, modelPath, dan,
                      sess, img_mask, img_size, conv_layer, logging=True)

    # if len(cam) > 0:
    #     cam_heatmap = get_heatmap(cam)
    # else:
    #     return None

    if vis_type == 'overlay':
        img = batch_img[0]
        overlay_cam(img, cam, alpha, show=True, outFile=outFile)


def get_most_activated_landmarks(
        cam, gt_landmarks, k, radius=3, img_size=224):
    """function that returns a number of landmarks that
    are most activated by gradcam

    for each gt landmark, activations in the area of radius r around landmark are summed
    (gt_x - r, gt_x + r)
    (gt_y - r, gt_y + r)

    k results are returned
    """
    landmark_activations = {}

    x_s = gt_landmarks[0:][::2]
    y_s = gt_landmarks[1:][::2]

    for x_, y_ in zip(x_s, y_s):
        x = min(int(x_), img_size - 1)
        y = min(int(y_), img_size - 1)

        landmark_activations[x, y] = 0
        # Add activations from around landmarks
        for r in range(radius):
            for s in range(radius):
                try:
                    landmark_activations[x, y] += cam[y + s][x + r]
                except IndexError:
                    print('')
    inverse = [(value, key) for key, value in landmark_activations.items()]
    sorted_inverse = sorted(inverse)[-k:]
    activated_landmarks = [b for _, b in sorted_inverse]
    return activated_landmarks
