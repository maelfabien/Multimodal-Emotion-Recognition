# coding=utf-8

import numpy as np
import math

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util.tf_export import tf_export


def e_distance(a, b):
    return np.linalg.norm(a - b)


def find_closest(matrix_list, target_matrix):
    min_distance = 999999
    closest_matrix = None
    closest_id = None
    for i, m in enumerate(matrix_list):
        if len(m) > 0:
            dist = e_distance(m, target_matrix)
            if dist < min_distance:
                closest_matrix = m
                closest_id = i
                min_distance = dist
    return closest_matrix, closest_id


def range_to_landmark_nb(landmark_list, landmark_dict):
    landmark_numbers = []
    for x,y in zip(landmark_list[0:][::2], landmark_list[1:][::2]):
        landmark_numbers.append(landmark_dict[int(x),int(y)])
    return landmark_numbers


def get_landmark_nb_dict(all_landmarks):
    xy_to_landmark = {}

    x_s = all_landmarks[0:][::2]
    y_s = all_landmarks[1:][::2]
    i_s = range(len(x_s))

    for i, x, y in zip(i_s, x_s, y_s):
        xy_to_landmark[int(x),int(y)] = i
        
    return xy_to_landmark

def loadFromPts(filename):
    landmarks = np.genfromtxt(filename, skip_header=3, skip_footer=1)
    landmarks = landmarks - 1

    return landmarks


def saveToPts(filename, landmarks):
    pts = landmarks + 1

    header = 'version: 1\nn_points: {}\n{{'.format(pts.shape[0])
    np.savetxt(filename, pts, delimiter=' ', header=header,
               footer='}', fmt='%.3f', comments='')


def bestFitRect(points, meanS, box=None):
    if box is None:
        box = np.array([points[:, 0].min(), points[:, 1].min(),
                        points[:, 0].max(), points[:, 1].max()])
    boxCenter = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    boxWidth = box[2] - box[0]
    boxHeight = box[3] - box[1]

    meanShapeWidth = meanS[:, 0].max() - meanS[:, 0].min()
    meanShapeHeight = meanS[:, 1].max() - meanS[:, 1].min()

    scaleWidth = boxWidth / meanShapeWidth
    scaleHeight = boxHeight / meanShapeHeight
    scale = (scaleWidth + scaleHeight) / 2

    S0 = meanS * scale

    S0Center = [(S0[:, 0].min() + S0[:, 0].max()) / 2,
                (S0[:, 1].min() + S0[:, 1].max()) / 2]
    S0 += boxCenter - S0Center

    return S0


def bestFit(destination, source, returnTransform=False):
    destMean = np.mean(destination, axis=0)
    srcMean = np.mean(source, axis=0)

    srcVec = (source - srcMean).flatten()
    destVec = (destination - destMean).flatten()

    a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec)**2
    b = 0
    for i in range(destination.shape[0]):
        b += srcVec[2 * i] * destVec[2 * i + 1] - \
            srcVec[2 * i + 1] * destVec[2 * i]
    b = b / np.linalg.norm(srcVec)**2

    T = np.array([[a, b], [-b, a]])
    srcMean = np.dot(srcMean, T)

    if returnTransform:
        return T, destMean - srcMean
    else:
        return np.dot(srcVec.reshape((-1, 2)), T) + destMean


def mirrorShape(shape, imgShape=None):
    imgShapeTemp = np.array(imgShape)
    shape2 = mirrorShapes(shape.reshape((1, -1, 2)),
                          imgShapeTemp.reshape((1, -1)))[0]

    return shape2


def mirrorShapes(shapes, imgShapes=None):
    shapes2 = shapes.copy()

    for i in range(shapes.shape[0]):
        if imgShapes is None:
            shapes2[i, :, 0] = -shapes2[i, :, 0]
        else:
            shapes2[i, :, 0] = -shapes2[i, :, 0] + imgShapes[i][1]

        lEyeIndU = list(range(36, 40))
        lEyeIndD = [40, 41]
        rEyeIndU = list(range(42, 46))
        rEyeIndD = [46, 47]
        lBrowInd = list(range(17, 22))
        rBrowInd = list(range(22, 27))

        uMouthInd = list(range(48, 55))
        dMouthInd = list(range(55, 60))
        uInnMouthInd = list(range(60, 65))
        dInnMouthInd = list(range(65, 68))
        noseInd = list(range(31, 36))
        beardInd = list(range(17))

        lEyeU = shapes2[i, lEyeIndU].copy()
        lEyeD = shapes2[i, lEyeIndD].copy()
        rEyeU = shapes2[i, rEyeIndU].copy()
        rEyeD = shapes2[i, rEyeIndD].copy()
        lBrow = shapes2[i, lBrowInd].copy()
        rBrow = shapes2[i, rBrowInd].copy()

        uMouth = shapes2[i, uMouthInd].copy()
        dMouth = shapes2[i, dMouthInd].copy()
        uInnMouth = shapes2[i, uInnMouthInd].copy()
        dInnMouth = shapes2[i, dInnMouthInd].copy()
        nose = shapes2[i, noseInd].copy()
        beard = shapes2[i, beardInd].copy()

        lEyeIndU.reverse()
        lEyeIndD.reverse()
        rEyeIndU.reverse()
        rEyeIndD.reverse()
        lBrowInd.reverse()
        rBrowInd.reverse()

        uMouthInd.reverse()
        dMouthInd.reverse()
        uInnMouthInd.reverse()
        dInnMouthInd.reverse()
        beardInd.reverse()
        noseInd.reverse()

        shapes2[i, rEyeIndU] = lEyeU
        shapes2[i, rEyeIndD] = lEyeD
        shapes2[i, lEyeIndU] = rEyeU
        shapes2[i, lEyeIndD] = rEyeD
        shapes2[i, rBrowInd] = lBrow
        shapes2[i, lBrowInd] = rBrow

        shapes2[i, uMouthInd] = uMouth
        shapes2[i, dMouthInd] = dMouth
        shapes2[i, uInnMouthInd] = uInnMouth
        shapes2[i, dInnMouthInd] = dInnMouth
        shapes2[i, noseInd] = nose
        shapes2[i, beardInd] = beard

    return shapes2


def cyclic_learning_rate(global_step,
                         learning_rate=0.01,
                         max_lr=0.1,
                         step_size=20.,
                         gamma=0.99994,
                         mode='triangular',
                         name=None):
    """Applies cyclic learning rate (CLR).
       From the paper:
       Smith, Leslie N. "Cyclical learning
       rates for training neural networks." 2017.
       [https://arxiv.org/pdf/1506.01186.pdf]
       This method lets the learning rate cyclically
       vary between reasonable boundary values
       achieving improved classification accuracy and
       often in fewer iterations.
       This code varies the learning rate linearly between the
       minimum (learning_rate) and the maximum (max_lr).
       It returns the cyclic learning rate. It is computed as:
        ```python
        cycle = floor( 1 + global_step /
          ( 2 * step_size ) )
        x = abs( global_step / step_size – 2 * cycle + 1 )
        clr = learning_rate +
          ( max_lr – learning_rate ) * max( 0 , 1 - x )
        ```
        Polices:
          'triangular':
            Default, linearly increasing then linearly decreasing the
            learning rate at each cycle.
          'triangular2':
            The same as the triangular policy except the learning
            rate difference is cut in half at the end of each cycle.
            This means the learning rate difference drops after each cycle.
          'exp_range':
            The learning rate varies between the minimum and maximum
            boundaries and each boundary value declines by an exponential
            factor of: gamma^global_step.
        Example: 'triangular2' mode cyclic learning rate.
          '''python
          ...
          global_step = tf.Variable(0, trainable=False)
          optimizer = tf.train.AdamOptimizer(learning_rate=
            clr.cyclic_learning_rate(global_step=global_step, mode='triangular2'))
          train_op = optimizer.minimize(loss_op, global_step=global_step)
          ...
          with tf.Session() as sess:
              sess.run(init)
              for step in range(1, num_steps+1):
                assign_op = global_step.assign(step)
                sess.run(assign_op)
          ...
          '''
        Args:
          global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
            Global step to use for the cyclic computation.  Must not be negative.
          learning_rate: A scalar `float32` or `float64` `Tensor` or a
          Python number.  The initial learning rate which is the lower bound
            of the cycle (default = 0.1).
          max_lr:  A scalar. The maximum learning rate boundary.
          step_size: A scalar. The number of iterations in half a cycle.
            The paper suggests step_size = 2-8 x training iterations in epoch.
          gamma: constant in 'exp_range' mode:
            gamma**(global_step)
          mode: one of {triangular, triangular2, exp_range}.
              Default 'triangular'.
              Values correspond to policies detailed above.
          name: String.  Optional name of the operation.  Defaults to
            'CyclicLearningRate'.
        Returns:
          A scalar `Tensor` of the same type as `learning_rate`.  The cyclic
          learning rate.
        Raises:
          ValueError: if `global_step` is not supplied.
       @compatibility(eager)
        When eager execution is enabled, this function returns
        a function which in turn returns the decayed learning
        rate Tensor. This can be useful for changing the learning
        rate value across different invocations of optimizer functions.
        @end_compatibility
    """

    if global_step is None:
        raise ValueError(
            "global_step is required for cyclic_learning_rate.")

    with ops.name_scope(name, "CyclicLearningRate",
                        [learning_rate, global_step]) as name:
        learning_rate = ops.convert_to_tensor(
            learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        step_size = math_ops.cast(step_size, dtype)

        def cyclic_lr():
            """Helper to recompute learning rate; most helpful in eager-mode."""
            # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
            double_step = math_ops.multiply(2., step_size)
            global_div_double_step = math_ops.divide(
                global_step, double_step)
            cycle = math_ops.floor(
                math_ops.add(
                    1., global_div_double_step))

            # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
            double_cycle = math_ops.multiply(2., cycle)
            global_div_step = math_ops.divide(global_step, step_size)
            tmp = math_ops.subtract(global_div_step, double_cycle)
            x = math_ops.abs(math_ops.add(1., tmp))

            # computing: clr = learning_rate + ( max_lr – learning_rate ) *
            # max( 0, 1 - x )
            a1 = math_ops.maximum(0., math_ops.subtract(1., x))
            a2 = math_ops.subtract(max_lr, learning_rate)
            clr = math_ops.multiply(a1, a2)

            if mode == 'triangular2':
                clr = math_ops.divide(clr, math_ops.cast(math_ops.pow(2, math_ops.cast(
                    cycle - 1, tf.int32)), tf.float32))
            if mode == 'exp_range':
                clr = math_ops.multiply(
                    math_ops.pow(gamma, global_step), clr)

            return math_ops.add(clr, learning_rate, name=name)

        if not context.executing_eagerly():
            cyclic_lr = cyclic_lr()

        return cyclic_lr
