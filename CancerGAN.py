import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time, cv2, os, pickle, random, math, pydicom
from statistics import mean
from timeit import default_timer as timer
from PIL import Image
#Get factors of a number below the cap if given
def factors(num, cap=None):
    retLst = []
    for i in range(1,num+1):
        if num%i == 0:
            retLst.append(i)
        if i == cap:
            break
    return retLst
#Slicing the images into a certain size
def imgSlice(img, xRes, yRes):
    slicedImg = []
    #img.show()
    width, height = img.size
    upper = 0
    slicesY = int(math.ceil(height/xRes))
    slicesX = int(math.ceil(width/yRes))

    countY = 1
    for sliceY in range(slicesY):
        #if we are at the end, set the lower bound to be the bottom of the image
        if countY == slicesY:
            lower = height
        else:
            lower = int(countY * yRes)  
        countX = 1
        left = 0
        for sliceX in range(slicesX):
            if countX == slicesX:
                right = width
            else:
                right = int(countX * yRes) 
            bbox = (left, upper, right, lower)
            working_slice = img.crop(bbox)
            #working_slice.show()
            slicedImg.append(working_slice)
            left += yRes
            countX += 1
        upper += xRes
        countY +=1
    return slicedImg
#Reading the images into an array that is usable
def setupData(imgX, imgY, categories, chunkX, chunkY, chunks, sizeOf=1):
    print("Loading Pictures.")
    dataDir = "C:/Users/Noah/Desktop/Cancer/"
    trainingData = []

    def createTrainingData(sizeOf):
        for category in categories:
            path = os.path.join(dataDir, category)
            classNum = categories.index(category)
            for img in os.listdir(path)[:int(len(os.listdir(path))/sizeOf)]:
                try:
                    chunkedData = []
                    imgArray = pydicom.dcmread(os.path.join(path, img))
                    imgArray = imgArray.pixel_array
                    resizeImg = cv2.resize(imgArray, (imgX, imgY))
                    #plt.imshow(resizeImg, cmap=plt.cm.bone)
                    #plt.show()
                    chunkOfImgs = imgSlice(Image.fromarray(resizeImg), chunkX, chunkY)
                    for i in chunkOfImgs:
                        chunkedData.append(np.asarray(i))
                    trainingData.append(chunkedData)
                except Exception as e:
                    print(e)

    createTrainingData(sizeOf)
    random.shuffle(trainingData)
    '''for i in trainingData[0]:
        plt.imshow(i)
        plt.show()'''

    images = []
    chunkedImgs = []

    for features in trainingData:
        for feature in features:
            images.append(feature)
    images = np.array(images).reshape(-1, chunkX, chunkY, 1)
    print("Pictures loaded.")
    return images
#Dicriminator
def makeDiscriminatorModel(x, y):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(7, (3, 3), padding="same", input_shape=(x, y, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(150, activation="relu"))
    model.add(tf.keras.layers.Dense(50, activation="relu"))
    model.add(tf.keras.layers.Dense(50, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    return model
#Making Discriminator
def getDiscriminatorLoss(realPredictions, fakePredictions):
    try:
        realPredictions = tf.sigmoid(realPredictions)
    except:
        pass
    fakePredictions = tf.sigmoid(fakePredictions)
    realLoss = tf.losses.binary_crossentropy(tf.ones_like(realPredictions), realPredictions)
    fakeLoss = tf.losses.binary_crossentropy(tf.zeros_like(fakePredictions), fakePredictions)
    return fakeLoss+realLoss
#Generator
def makeGeneratorModel(x, y):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(int(x/4)*int(y/4)*256, input_shape=(latentSize,)))#7*7*256 for 28x28 img and 14*14*256 for 56x56 image so 1 for a 4x4 img
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape((int(x/4), int(y/4), 256)))
    model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2,2), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2,2), padding="same", activation="relu"))
    return model
#Making Generator
def getGeneratorLoss(fakePredictions):
    fakePredictions = tf.sigmoid(fakePredictions)
    fakeLoss = tf.losses.binary_crossentropy(tf.ones_like(fakePredictions), fakePredictions)
    return fakeLoss
#Training
def train(dataset, epochs, chunks, noise):
    print("Training Started.")
    for chunk in range(chunks):
        cStart = timer()
        for i in range(epochs):
            epochDiscLoss.clear()
            epochGenLoss.clear()
            epochDiscLoss.append(0)
            epochGenLoss.append(2)
            start = timer()
            count = 0
            latentPoint = 0
            imgList = []
            for images in dataset:
                count += 1
                imgList.append(images)
                if count == chunks:
                    #images = tf.cast(images, tf.dtypes.float32)
                    trainStep(imgList, chunk, np.random.randn(BATCH_SIZE, latentSize))
                    latentPoint += 1
                    count = 0
                    imgList.clear()
            end = timer()
            if ((epochs-(i+1))*(end-start)) > (60*60):
                print(f"Time Left: {int(((epochs-(i+1))*(end-start))//3600)}:{int((((epochs-(i+1))*(end-start))//60)-((((epochs-(i+1))*(end-start))//3600)*60))}:{int(((epochs-(i+1))*(end-start))%60)}")
            else:
                print(f"Time Left: {int(((epochs-(i+1))*(end-start))//60)} min {int(((epochs-(i+1))*(end-start))%60)} sec")
            print(f"Epoch {i+1} at {chunkX}x{chunkY} finished in {end-start} seconds.")
            print(f"The mean generator loss for epoch {i+1} is {mean(epochGenLoss)}.")
            print(f"The mean discriminator loss for epoch {i+1} is {mean(epochDiscLoss)}.\n")
            #epochs += 1
        cEnd = timer()
        if cEnd-cStart > 60:
            print(f"Chunk {chunk+1} finished in {int((cEnd-cStart)//60)} minutes and {(cEnd-cStart)%60} seconds.")
        else:
            print(f"Chunk {chunk+1} finished in {cEnd-cStart} seconds.")
        print(f"The total mean generator loss for chunk {chunk+1} is {mean(totalGenLoss)}.")
        print(f"The total mean discriminator loss for chunk {chunk+1} is {mean(totalDiscLoss)}.\n\n")
    return epochs
#Getting gradients and applying the corrections to the network
def trainStep(images, chunk, fakeImageNoise):
    #fakeImageNoise = np.random.randn(BATCH_SIZE, latentSize)
    #print("Chunk: "+str(chunk))
    with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
        generatedImages = generators[chunk](fakeImageNoise)
        realOutput = discriminators[chunk](images[chunk])
        fakeOutput = discriminators[chunk](generatedImages)
        #Getting gradient or loss
        genLoss = getGeneratorLoss(fakeOutput)
        discLoss = getDiscriminatorLoss(realOutput, fakeOutput)
        #Find gradient
        gradientsOfGen = genTape.gradient(genLoss, generators[chunk].trainable_variables)
        gradientsOfDisc = discTape.gradient(discLoss, discriminators[chunk].trainable_variables)
        #Using optimizer
        generatorOptimizer.apply_gradients(zip(gradientsOfGen, generators[chunk].trainable_variables))
        discriminatorOptimizer.apply_gradients(zip(gradientsOfDisc, discriminators[chunk].trainable_variables))

        epochDiscLoss.append(np.mean(discLoss))
        epochGenLoss.append(np.mean(genLoss))
        totalDiscLoss.append(np.mean(discLoss))
        totalGenLoss.append(np.mean(genLoss))
        #print("Generator Loss: ", np.mean(genLoss))
        #print("Discriminator Loss: ", np.mean(discLoss))
#Constants
xRes = 256
yRes = 256
chunkX = 256
chunkY = 256
chunks = int((xRes/chunkX)**2)
if chunks == 2:
    chunks = 1
cancer1 = "ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-003/12-18-1959-CT THORAX WITH CONTRAS-44882/3-Recon 2 HELICAL-29189"
cancer2 = "ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-001/01-10-1960-Abdomen1WBPETCT-36994/2-Abd.CT 5.0 B30s-63370"
lungCancer = "ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-003/12-25-1959-NSC LUNG RESTG-84053/2-CT Atten Cor Head In-69694"
axialLung = "ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-003/05-19-1960-Thorax1CHESTWITH Adult-60629/3-AXIAL LUNG-55609"
epochs = 75000
epochs = 1000
latentSize = 100
epochGenLoss = [2]
epochDiscLoss = [0]
totalGenLoss = [2]
totalDiscLoss = [0]
dataset = [[cancer1], [cancer2], [lungCancer], [axialLung]]
#Data Specific Constants
trainImages = setupData(xRes, yRes, dataset[3], chunkX, chunkY, chunks)
trainImages = trainImages/255.0
trainImages = trainImages.reshape(trainImages.shape[0], chunkX, chunkY, 1)
BUFFER_SIZE = trainImages.shape[1]
BATCH_SIZE = factors(int(trainImages.shape[0]/chunks), cap=50)[-1]
BATCH_SIZE = BATCH_SIZE*chunks
trainDataset = tf.data.Dataset.from_tensor_slices(trainImages).batch(BATCH_SIZE)
print(f"\n\nThe Dataset size is: {int(trainImages.shape[0]/chunks)}\n\nThe Chunk size is {chunkX}x{chunkY} with {chunks} chunks\n\nThere are {trainImages.shape[0]} total chunked images\n\nThe BATCH_SIZE is: {BATCH_SIZE}\n\n")
#Creating the latent arrays
allLatentPoints = []
#for i in range(int(((trainImages.shape[0]/chunks)/(BATCH_SIZE/chunks))*epochs)):
    #allLatentPoints.append(np.random.randn(BATCH_SIZE, latentSize))
#Making Dicsriminator & Generator
discriminators = []
generators = []
for chunk in range(chunks):#Not enough vram to hold all models
    discriminators.append(makeDiscriminatorModel(chunkX, chunkY))
    generators.append(makeGeneratorModel(chunkX, chunkY))
discriminatorOptimizer = tf.optimizers.Adam(1e-5)
generatorOptimizer = tf.optimizers.Adam(1e-6)
#overallDisc = makeDiscriminatorModel(xRes, yRes)
#Main Loop
fullStart = timer()
epochs = train(trainDataset, epochs, chunks, allLatentPoints)
fullStop = timer()
if (fullStop-fullStart) > (60*60):
    print(f"Time Left: {int((fullStop-fullStart)//3600)}:{int(((fullStop-fullStart)//60)-(((fullStop-fullStart)//3600)*60))}:{int((fullStop-fullStart)%60)}")
elif fullStop-fullStart > 60:
    print(f"It took {int((fullStop-fullStart)//60)} minutes and {int((fullStop- fullStart)%60)} seconds to finish {epochs} epochs.")
else:
    print(f"It took {int(fullStop-fullStart)} seconds to finish {epochs} epochs.")
again = input("Press Enter to see the results: ")
while again == "":
    for chunk in range(chunks):
        plt.imshow(tf.reshape(generators[chunk](np.random.randn(1, latentSize)), (chunkX, chunkY)), cmap=plt.cm.bone)
        plt.show()
    again = input("Press Enter to see the results: ")
for chunk in range(chunks):
    generators[chunk].save(f"D:\ML\\GAN\\ganModels\\Cancer\\chunk{chunk+1}cancerGENModel{xRes}x{yRes}res-{epochs}epochs-{latentSize}latent.model")
    discriminators[chunk].save(f"D:\ML\\GAN\\ganModels\\Cancer\\chunk{chunk+1}cancerDISCModel{xRes}x{yRes}res-{epochs}epochs-{latentSize}latent.model")