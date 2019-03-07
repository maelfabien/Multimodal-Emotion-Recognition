"""
Trying to make data loading more efficient.

!python3.6 train.py "/home/workspace/itautkute/emoDAN-tensorflow/" --trainSetFile="AffectnetVal_7.npz" 2
!CUDA_VISIBLE_DEVICES=1 python3.6 train.py "/home/itautkute/emotionaldan/data/"   2

"""

import argparse
import numpy as np
import tensorflow as tf

from models import emoDAN

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('datasetDir')
    parser.add_argument('stage', type=int)
    parser.add_argument('--trainSetFile', default='AffectnetTrain_7.npz')
    parser.add_argument(
        '--validationSetFile',
        default='AffectnetVal_7.npz')
    parser.add_argument('--imageHeight', default=224)
    parser.add_argument('--imageWidth', default=224)
    parser.add_argument('--valSetSize', default=200)
    parser.add_argument('--batchSize', default=50)
    parser.add_argument('--epochs', default=40)
    parser.add_argument('--pretrainedModel', default='./Model0/Model')
    parser.add_argument('--outPath', default='./Model2/Model2')
    

    args = parser.parse_args()

    trainSet = np.load(args.datasetDir + args.trainSetFile)
    validationSet = np.load(args.datasetDir + args.validationSetFile)

    Xtrain = trainSet['Image']
    Ytrain = trainSet['Landmark']
    Ytrain_em = trainSet['Emotion']

    Xvalid = validationSet['Image'][:args.valSetSize]
    Yvalid = validationSet['Landmark'][:args.valSetSize]
    Yvalid_em = validationSet['Emotion'][:args.valSetSize]

    nChannels = Xtrain.shape[3]
    nSamples = Xtrain.shape[0]
    testIdxsTrainSet = range(len(Xvalid))
    testIdxsValidSet = range(len(Xvalid))

    meanImg = trainSet['MeanShape']
    initLandmarks = trainSet['Landmark'][0].reshape((1, 136))

    with tf.Session() as sess:
        emotionaldan = emoDAN(initLandmarks, args.batchSize)

        Saver = tf.train.Saver()
        Writer = tf.summary.FileWriter("logs/", sess.graph)

        if args.stage < 2:
            sess.run(tf.global_variables_initializer())
        else:
            Saver.restore(sess, args.pretrainedModel)
            print('Pre-trained model has been loaded!')

        print("Starting training......")
        max_accuracy = 0
        min_err = 99999
        global_step = tf.Variable(0, trainable=False)

        for epoch in range(args.epochs):

            Count = 0
            while Count * args.batchSize < Xtrain.shape[0]:

                RandomIdx = np.random.choice(Xtrain.shape[0],args.batchSize,False)

                if args.stage == 1 or args.stage == 0:
                    # Training landmarks
                    sess.run(
                        emotionaldan['S1_Optimizer'],
                        feed_dict={emotionaldan['InputImage']: Xtrain[RandomIdx],
                                   emotionaldan['GroundTruth']: Ytrain[RandomIdx],
                                   emotionaldan['Emotion_labels']: Ytrain_em[RandomIdx],
                                   emotionaldan['S1_isTrain']: True,
                                   emotionaldan['S2_isTrain']: False,
                                   # emotionaldan['lr_stage2']:learning_rate
                                   })
                else:
                    # Training emotions
                    sess.run(
                        # emotionaldan['S2_Optimizer'],
                        # feed_dict={emotionaldan['InputImage']:Xtrain[RandomIdx],
                        #            emotionaldan['GroundTruth']:Ytrain[RandomIdx],
                        #            emotionaldan['Emotion_labels']:Ytrain_em[RandomIdx],
                        #            emotionaldan['S1_isTrain']:False,
                        #            emotionaldan['S2_isTrain']:True,
                        #            # emotionaldan['lr_stage2']:learning_rate
                        #           })
                        [emotionaldan['S2_Optimizer'],
                            emotionaldan['iterator'].initializer],
                        feed_dict={emotionaldan['x']: Xtrain,
                                   emotionaldan['y']: Ytrain,
                                   emotionaldan['z']: Ytrain_em,
                                   emotionaldan['S1_isTrain']: False,
                                   emotionaldan['S2_isTrain']: True,
                                   # emotionaldan['lr_stage2']:learning_rate
                                   })

                if Count % 256 == 0:
                    TestErr = 0
                    BatchErr = 0

                    if args.stage == 1 or args.stage == 0:
                        # Validation landmarks
                        TestErr = sess.run(
                            emotionaldan['S1_Cost'],
                            {emotionaldan['InputImage']: Xvalid,
                             emotionaldan['GroundTruth']: Yvalid,
                             emotionaldan['Emotion_labels']: Yvalid_em,
                             emotionaldan['S1_isTrain']: False,
                             emotionaldan['S2_isTrain']: False,
                             # emotionaldan['lr_stage2']:learning_rate
                             })
                        BatchErr = sess.run(
                            emotionaldan['S1_Cost'],
                            {emotionaldan['InputImage']: Xtrain[RandomIdx],
                             emotionaldan['GroundTruth']: Ytrain[RandomIdx],
                             emotionaldan['Emotion_labels']: Ytrain_em[RandomIdx],
                             emotionaldan['S1_isTrain']: False,
                             emotionaldan['S2_isTrain']: False,
                             # emotionaldan['lr_stage2']:learning_rate
                             })
                        print('Epoch: ', epoch, ' Batch: ', Count,
                              'TestErr:', TestErr, ' BatchErr:', BatchErr)
                        if TestErr < min_err:
                            Saver.save(sess, args.outPath)
                            min_err = TestErr
                    else:
                        # Validation emotions
                        # TestErr, accuracy_test = sess.run(
                        #     [emotionaldan['Joint_Cost'], emotionaldan['Emotion_Accuracy']],
                        #     {emotionaldan['InputImage']:Xvalid,
                        #      emotionaldan['GroundTruth']:Yvalid,
                        #      emotionaldan['Emotion_labels']:Yvalid_em,
                        #      emotionaldan['S1_isTrain']:False,
                        #      emotionaldan['S2_isTrain']:False})
                        TestErr, accuracy_test = sess.run(
                            [emotionaldan['Joint_Cost'],
                                emotionaldan['Emotion_Accuracy']],
                            {emotionaldan['x']: Xvalid,
                             emotionaldan['y']: Yvalid,
                             emotionaldan['z']: Yvalid_em,
                             emotionaldan['S1_isTrain']: False,
                             emotionaldan['S2_isTrain']: False})
                        # BatchErr, accuracy_train, learn_rate = sess.run(
                        #     [emotionaldan['Joint_Cost'], emotionaldan['Emotion_Accuracy'], emotionaldan['lr']],
                        #     {emotionaldan['InputImage']:Xtrain[RandomIdx],
                        #      emotionaldan['GroundTruth']:Ytrain[RandomIdx],
                        #      emotionaldan['Emotion_labels']:Ytrain_em[RandomIdx],
                        #      emotionaldan['S1_isTrain']:False,
                        #      emotionaldan['S2_isTrain']:False})
                        BatchErr, accuracy_train, learn_rate = sess.run(
                            [emotionaldan['Joint_Cost'],
                                emotionaldan['Emotion_Accuracy'], emotionaldan['lr']],
                            {emotionaldan['x']: Xtrain,
                             emotionaldan['y']: Ytrain,
                             emotionaldan['z']: Ytrain_em,
                             emotionaldan['S1_isTrain']: False,
                             emotionaldan['S2_isTrain']: False})
                        announce = 'Epoch: ' + str(epoch) + ' Batch: ' + str(Count) + ' TestErr: ' + str(TestErr) + ' BatchErr: ' + str(
                            BatchErr) + ' TestAcc: ' + str(accuracy_test) + ' TrainAcc: ' + str(accuracy_train) + ' LR: ' + str(learn_rate) + '\n'
                        print(announce)
                        with open('logging.txt', 'a') as my_file:
                            my_file.write(announce)
                        if accuracy_test > max_accuracy:
                            Saver.save(sess, args.outPath)
                            max_accuracy = accuracy_test
                Count += 1
