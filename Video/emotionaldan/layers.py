# Courtesy of https://github.com/mariolew/Deep-Alignment-Network-tensorflow

import tensorflow as tf
import numpy as np
import itertools

IMGSIZE = 224
N_LANDMARK = 68

Pixels = tf.constant(np.array([(x, y) for x in range(IMGSIZE) for y in range(IMGSIZE)], dtype=np.float32),
                     shape=[IMGSIZE, IMGSIZE, 2])


def TransformParamsLayer(SrcShapes, DstShape):
    '''
    SrcShapes: [N, (N_LANDMARK x 2)]
    DstShape: [N_LANDMARK x 2,]
    return: [N, 6]
    '''
    # import pdb; pdb.set_trace()
    def bestFit(src, dst):
        # import pdb; pdb.set_trace()
        source = tf.reshape(src, (-1, 2))
        destination = tf.reshape(dst, (-1, 2))

        destMean = tf.reduce_mean(destination, axis=0)
        srcMean = tf.reduce_mean(source, axis=0)

        srcCenter = source - srcMean
        dstCenter = destination - destMean

        srcVec = tf.reshape(srcCenter, (-1,))
        destVec = tf.reshape(dstCenter, (-1,))
        norm = (tf.norm(srcVec)**2)

        a = tf.tensordot(srcVec, destVec, 1) / norm
        b = 0

        srcX = tf.reshape(srcVec, (-1, 2))[:, 0]
        srcY = tf.reshape(srcVec, (-1, 2))[:, 1]
        destX = tf.reshape(destVec, (-1, 2))[:, 0]
        destY = tf.reshape(destVec, (-1, 2))[:, 1]

        b = tf.reduce_sum(
            tf.multiply(
                srcX,
                destY) -
            tf.multiply(
                srcY,
                destX))
        b = b / norm

        A = tf.reshape(tf.stack([a, b, -b, a]), (2, 2))
        srcMean = tf.tensordot(srcMean, A, 1)

        return tf.concat((tf.reshape(A, (-1,)), destMean - srcMean), 0)

    return tf.map_fn(lambda s: bestFit(s, DstShape), SrcShapes)


def AffineTransformLayer(Image, Param):
    '''
    Image: [N, IMGSIZE, IMGSIZE, 2]
    Param: [N, 6]
    return: [N, IMGSIZE, IMGSIZE, 2]
    '''

    A = tf.reshape(Param[:, 0:4], (-1, 2, 2))
    T = tf.reshape(Param[:, 4:6], (-1, 1, 2))

    A = tf.matrix_inverse(A)
    T = tf.matmul(-T, A)

    T = tf.reverse(T, (-1,))
    A = tf.matrix_transpose(A)

    def affine_transform(I, A, T):
        I = tf.reshape(I, [IMGSIZE, IMGSIZE])

        SrcPixels = tf.matmul(tf.reshape(
            Pixels, [IMGSIZE * IMGSIZE, 2]), A) + T
        SrcPixels = tf.clip_by_value(SrcPixels, 0, IMGSIZE - 2)

        outPixelsMinMin = tf.to_float(tf.to_int32(SrcPixels))
        dxdy = SrcPixels - outPixelsMinMin
        dx = dxdy[:, 0]
        dy = dxdy[:, 1]

        outPixelsMinMin = tf.reshape(tf.to_int32(
            outPixelsMinMin), [IMGSIZE * IMGSIZE, 2])
        outPixelsMaxMin = tf.reshape(
            outPixelsMinMin + [1, 0], [IMGSIZE * IMGSIZE, 2])
        outPixelsMinMax = tf.reshape(
            outPixelsMinMin + [0, 1], [IMGSIZE * IMGSIZE, 2])
        outPixelsMaxMax = tf.reshape(
            outPixelsMinMin + [1, 1], [IMGSIZE * IMGSIZE, 2])

        OutImage = (1 - dx) * (1 - dy) * tf.gather_nd(I, outPixelsMinMin) + dx * (1 - dy) * tf.gather_nd(I, outPixelsMaxMin) \
            + (1 - dx) * dy * tf.gather_nd(I, outPixelsMinMax) + \
            dx * dy * tf.gather_nd(I, outPixelsMaxMax)

        return tf.reshape(OutImage, [IMGSIZE, IMGSIZE, 1])

    return tf.map_fn(lambda args: affine_transform(
        args[0], args[1], args[2]), (Image, A, T), dtype=tf.float32)


def LandmarkTransformLayer(Landmark, Param, Inverse=False, nb_landmarks=N_LANDMARK):
    '''
    Landmark: [N, N_LANDMARK x 2]
    Param: [N, 6]
    return: [N, N_LANDMARK x 2]
    '''

    A = tf.reshape(Param[:, 0:4], [-1, 2, 2])
    T = tf.reshape(Param[:, 4:6], [-1, 1, 2])

    Landmark = tf.reshape(Landmark, [-1, nb_landmarks, 2])
    if Inverse:
        A = tf.matrix_inverse(A)
        T = tf.matmul(-T, A)

    return tf.reshape(tf.matmul(Landmark, A) + T, [-1, nb_landmarks * 2])


HalfSize = 8

Offsets = tf.constant(np.array(list(itertools.product(range(-HalfSize, HalfSize),
                                                      range(-HalfSize, HalfSize))), dtype=np.int32), shape=(16, 16, 2))


def LandmarkImageLayer(Landmarks):

    def draw_landmarks(L):
        def draw_landmarks_helper(Point):
            intLandmark = tf.to_int32(Point)
            locations = Offsets + intLandmark
            dxdy = Point - tf.to_float(intLandmark)
            offsetsSubPix = tf.to_float(Offsets) - dxdy
            vals = 1 / (1 + tf.norm(offsetsSubPix, axis=2))
            img = tf.scatter_nd(locations, vals, shape=(IMGSIZE, IMGSIZE))
            return img
        Landmark = tf.reverse(tf.reshape(L, [-1, 2]), [-1])
        # Landmark = tf.reshape(L, (-1, 2))
        Landmark = tf.clip_by_value(
            Landmark, HalfSize, IMGSIZE - 1 - HalfSize)
        # Ret = 1 / (tf.norm(tf.map_fn(DoIn,Landmarks),axis = 3) + 1)
        Ret = tf.map_fn(draw_landmarks_helper, Landmark)
        Ret = tf.reshape(tf.reduce_max(Ret, axis=0), [IMGSIZE, IMGSIZE, 1])
        return Ret
    return tf.map_fn(draw_landmarks, Landmarks)


def GetHeatMap(Landmark):

    def Do(L):
        def DoIn(Point):
            return Pixels - Point
        Landmarks = tf.reverse(tf.reshape(L, [-1, 2]), [-1])
        Landmarks = tf.clip_by_value(
            Landmarks, HalfSize, 112 - 1 - HalfSize)
        Ret = 1 / (tf.norm(tf.map_fn(DoIn, Landmarks), axis=3) + 1)
        Ret = tf.reshape(tf.reduce_max(Ret, 0), [IMGSIZE, IMGSIZE, 1])
        return Ret
    return tf.map_fn(Do, Landmark)
