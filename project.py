## Christopher Chan, Kim Le, Thomas Pham, Hyun Guk Yoo
## ccc180002@utdallas.edu, kml190002@utdallas.edu, ttp190005@utdallas.edu, hxy170002@utdallas.edu

import numpy as np
import pandas as pd
from PIL import Image
import requests
import os
from typing import List, Self
from matplotlib import pyplot as plt
import time
import random

## Implementation for Convolution Layer. 
class Conv3x3():

  ## The filters to be optimized are randomly initialized 
  ## when the object is created
  def __init__(self, numFilters) -> None:
    self.numFilters = numFilters

    ## k number of 3x3 Filters
    #np.random.seed(SEED_NUMBER)
    self.filters = np.random.randn(numFilters, 3, 3) / 9

  ## This iterates through the 2x2 dimension and generates the valid 3x3 image region 
  ## and stores a 3x3 image region for each index
  ## Image is 2D array (height, width)
  def iterateRegions(
    self,
    image: np.array,
  ):
    imageHeight, imageWidth = image.shape

  ## regionMap is the new mapping of the filter from the image to
  ## to convolution layer
    regionMap = [[0 for col in range(imageWidth - 2)] for row in range(imageHeight -2)]

    for col in range(imageHeight - 2):
      for row in range(imageWidth - 2):
        imageRegion = image[col:(col+3), row:(row+3)]
        regionMap[row][col] = imageRegion
        
    return regionMap
  

  ## This is the iterate regions for the backward pass
  def backPassRegions(
      self,
      image: np.array):
    ##This actaully produces a region of an image that also captures its DEPTH
    try:
      imageHeight, imageWidth, _ = image.shape
    except:
      imageHeight, imageWidth = image.shape

    for col in range(imageHeight - 2):
      for row in range(imageWidth - 2):
        imageRegion = image[col:(col + 3), row:(row + 3)]
        yield imageRegion, col, row #will return the imageRegion for every index col, row 
    
  

  ## Forward Pass Implementation
  ## Input is 3D numpy array (Height, Width, NumFilters)
  ## So if input is 148 x 148 x 8, it means 8 filters of size 148 x 148
  ## This is an example of what it does to the dimensions. 
  ## The first iteration receives a 2D 150 x 150 and turns it to 3D 8 x 148 x 148
  ## Every iteration after will recieve a 3D input and return 3D input. Here is example.
  ## Recieves input from MaxPool and returns Conv3x3 Layer.
  ## Pool1: (8, 74, 74)
  ## Conv2: (16, 72, 72)
  ## Doubles the filters number and reduces the dimension by 2 of x and y because this
  ## Takes a 3x3 region, compute dot product with filter and sums new 3x3 matrix. This will 
  ## be mapped into only one index of the new layer.
  def forwardPass(
    self,
    input: np.ndarray,
    firstPass: bool
  ):
    ## Stores the previous input coming into it. This is used to reconstruct dimensions
    ## and new image to be used during the backward pass
    self.lastInput = input

    ## First Pass so there is only one layer
    if firstPass == True:
      Matrix3D = self.iterateRegions(input)
      height = len(Matrix3D)
      width = len(Matrix3D[0])

      ## output = 1 filter 2D matrix
      # output = [[0 for col in range(width)] for row in range(height)]
      ## reuslt = all filters so 3D matrix
      result = []
      depth = 1
      
      currentDepth = Matrix3D
      for filter in range(0, self.numFilters):
        output = [[0 for col in range(width)] for row in range(height)]
        currentFilter = self.filters[filter]
        for col in range(height):
          for row in range(width):
            currentRegion = currentDepth[row][col]
            output[row][col] = np.sum(np.dot(np.array(currentRegion), np.array(currentFilter)))
        
        result.append(output)

      npArray = np.array(result)
      return npArray
    
    else:
      Input3D = input
      numLayers = len(input)
      height = len(input[0])
      width = len(input[0][0])

      CombinedInput = [[0 for col in range(height)] for row in range(width)]
      
      for col in range(0, height):
        for row in range(0, width):
          combinedLayerAtIndex = 0
          for layer in range(0, numLayers):
            combinedLayerAtIndex += Input3D[layer][col][row] 
          CombinedInput[row][col] = combinedLayerAtIndex

      CombinedInputNpArray = np.array(CombinedInput)
      Matrix3D = self.iterateRegions(CombinedInputNpArray)
      Matrix3DNpArray = np.array(Matrix3D)
  
  
        ## result = all filters so 3D matrix
      result = []

      for filter in range(0, self.numFilters):
        output = [[0 for col in range(width - 2)] for row in range(height -2)]

        currentLayer = Matrix3D
        currentFilter = self.filters[filter]
        for col in range(height-2):
          for row in range(width-2):
            currentRegion = currentLayer[row][col]
            output[row][col] = np.sum(np.dot(np.array(currentRegion), np.array(currentFilter)))
        
        result.append(output)

      npArray = np.array(result)
      return npArray
    

  ## Backward pass for Conv3x3.
  ## Updates the filters with teh gradients and reconstsructs the Convolution Layer
  ## Conv4: (64, 15, 15)
  ## Pool3: (32, 17, 17)

  def backward(
      self, 
      d_L_d_out: np.ndarray, 
      learn_rate: float, 
      first: bool
      ):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        d_L_d_filters = np.zeros(self.filters.shape)
        #d_L_d_filters = np.zeros((self.filters.shape[0], self.lastInput.shape[1], self.lastInput.shape[1]))
        d_L_d_input = np.zeros(self.lastInput.shape)  # Gradient w.r.t input
        d_L_d_out_padded = np.pad(d_L_d_out, [(0,0), (1,1), (1,1)], mode='constant', constant_values=0)

        for im_region, i, j in self.backPassRegions(self.lastInput):
            if i >= d_L_d_input.shape[1]-2 or j >= d_L_d_input.shape[1]-2 or i == 0 or j == 0:
                continue
            for f in range(self.numFilters):
                
                ## If you are at the final layer of backward pass, so you are back to original image
                if first == True:
                  d_L_d_filters[f] += np.sum(np.dot(d_L_d_out_padded[f, i, j], im_region))

                  try:    
                    d_L_d_input[:,i:i+3, j:j+3] += d_L_d_out_padded[f,i, j] * self.filters[f]
                  except:
                    pass

                else:
                  d_L_d_filters[f] += np.sum(np.dot(d_L_d_out_padded[f, i, j], im_region), axis = 2)
                  try:
                    d_L_d_input[:,i:i+3, j:j+3] += d_L_d_out_padded[f,i, j] * self.filters[f]
                  except:
                    pass

        # Update filters
        self.filters -= learn_rate * d_L_d_filters
        # Return the loss gradient for this layer's inputs
        return d_L_d_input



## Implementation of MaxPool Class.
class MaxPool2:


  ## Image is 2D array (height, width)
  def iterateRegions(
    self, 
    image: np.array
  ):
    numFilters, imageHeight, imageWidth = image.shape
    newHeight = imageHeight // 2
    newWidth = imageWidth // 2


    result = []
    for filter in range(0,numFilters):
      currentDepth = image[filter]
      regionMap = [[0 for col in range(newHeight)] for row in range(newWidth)]

      for col in range(newHeight):
        for row in range(newWidth):
          colMin = col * 2
          colMax = col * 2 + 2
          rowMin = row * 2
          rowMax = row * 2 + 2
          imageRegion = currentDepth[colMin:colMax, rowMin:rowMax]
          regionMap[row][col] = imageRegion
          
      result.append(regionMap)
    return result
  

  ## Forward Pass Implementation of MaxPool
  ## Recieves as input from Conv3x3 Layer. The finds the max of the 2x2 region 
  ## and halves the dimension of the original Conv3x3 layer coming into it
  ## Conv1: (8, 148, 148)
  ## Pool1: (8, 74, 74)
  ## This works by taking the max of the 2x2 region and taking only the max value of it
  ## and mapping it into the new output 
  def forward(
    self,
    input: np.array,
  ):
    ## Reduced Matrix by 2 with 2x2 image regions
    ## So Matrix 3D is (filter, height, width, dimensions)
    ## (8, 74, 74), so 8 filters of 74x74 dimensions with each index being a 2x2 image region
    Matrix3D = self.iterateRegions(input)
    numFilters = len(Matrix3D)
    height = len(Matrix3D[0])
    width = len(Matrix3D[0][0])

    result = []
    
    for filter in range(0, numFilters):
      currentLayer = Matrix3D[filter]
      output = [[0 for col in range(width)] for row in range(height)]

      for col in range(height):
        for row in range(width):
          currentRegion = currentLayer[row][col]
          output[row][col] = np.max(currentRegion)

      result.append(output)
    

    ## Stores the previous input coming into it. This is used to reconstruct dimensions
    ## and new image to be used during the backward pass
    npArray = np.array(result)
    self.previousInput = input
    self.previousMatrix3D = Matrix3D

    return npArray
  

  ## Backward pass of MaxPool implementation
  ## This takes uses the input saved from forward pass. 
  ## So during forward pass a region could look like this and look like this
  ## after the reconstruction of the backward pass.
  ## it only takes the max of that 2x2 region and replaces that index
  ## with the gradient and everything else iwll be zero 
  ## grad = gradient. It is recieved as input from output of softMax gradient
  ## [[0 ,55, 0 ,0],       [[0, grad, 0, 0],
  ## [20 ,0, 41 ,33],  ->  [0, 0, grad, 0],
  ## [0 ,90, 0 ,0],        [0, grad, 0, 0],
  ## [0 ,57, 0 ,95]]       [0, 0, 0, grad]]
  ## In doing the implementaiton of this, we would be restoring the original 
  ## dimensions before coming into the pool so that is why the dimensions are
  ## doubled again, because we are mapping the region and doubling it.
  ## Pool4: (64, 7, 7)
  ## Conv4: (64, 15, 15)
  def backward(
    self, 
    d_L_d_out: np.ndarray
  ):
    numLayers = len(self.previousInput)
    height = len(self.previousInput[0])
    width = len(self.previousInput[0][0])

    d_L_d_input = []

    for layer in range(0, numLayers):
      output = [[0 for col in range(width)] for row in range(height)]

      for row in range(height): 
        if height % 2 != 0 and row == height -1:
          continue
        for col in range(width):
          if width % 2 != 0 and col == width -1:
            continue
          if row % 2 != 0:
            continue
          if col % 2 != 0:
            continue
          else:
            currentRegion = [self.previousInput[layer][row][col],self.previousInput[layer][row][col+1],self.previousInput[layer][row+1][col],self.previousInput[layer][row+1][col+1]]
            regionMax = max(currentRegion)
            maxIndex = currentRegion.index(regionMax)
            if maxIndex == 0:
              currMaxIndexX = col
              currMaxIndexY = row
            elif maxIndex == 1:
              currMaxIndexX = col + 1
              currMaxIndexY = row 
            elif maxIndex == 2:
              currMaxIndexX = col 
              currMaxIndexY = row + 1
            else:
              currMaxIndexX = col + 1
              currMaxIndexY = row + 1

            output[currMaxIndexY][currMaxIndexX] = d_L_d_out[layer][row // 2][col // 2]

      d_L_d_input.append(output)

    npArrayd_L_d_input = np.array(d_L_d_input)
    return npArrayd_L_d_input

  

## Implementation of SoftMax Class
class Softmax:
  # A standard fully-connected layer with softmax activation.

  def __init__(self, input_len, nodes):
    # We divide by input_len to reduce the variance of our initial values
    #np.random.seed(SEED_NUMBER)
    self.weights = np.random.randn(input_len, nodes) / input_len
    self.biases = np.zeros(nodes)

  ## Recieves input from the final pool layer and the flattens the input
  ## into a 1D vector. This vector then computes the dot product of the 1D
  ## Vector the weights and biases to be ready to make the prediction.
  ## Then returns a matrix with the probability of each label in that index
  ## The max of this probability matrix is the prediction label for that image
  ## Returns a 1d numpy array containing the respective probability values.
  ## input can be any array with any dimensions.
  ## Pool4: (64, 7, 7)
  ## SoftMax:(3136, 1)
  ## Prediction: [.1, .1, .1, .1, .1, .5] So the predicted label would be 6
  def forward(
    self, 
    input: np.array,
    weights: np.ndarray,
    biases: np.ndarray,
  ):
    
    
    
    inputFlattened = input.flatten()  

    totals = np.dot(inputFlattened, weights) + biases

    ## Stores the previous input coming into it. This is used to reconstruct dimensions
    ## and new image to be used during the backward pass
    self.lastInputShape = input.shape
    self.lastInputFlattened = inputFlattened
    self.lastTotals = totals

    exp = np.exp(totals)

    return exp / np.sum(exp, axis=0)

  ## Backward pass for softmax
  ## Compute the gradient for the softmax layer and returns the gradient to 
  ## To be used as input for the last maxpool layer
  ## SoftMax:(3136, 1)
  ## Pool4: (64, 7, 7)
  def backwardPass(
    self, 
    weights: np.ndarray,
    biases: np.ndarray, 
    initialGradient: np.ndarray, 
    learningRate: float,
    ):

    for i, gradient in enumerate(initialGradient):
      if gradient == 0:
        continue

      # e^total[i] 
      totalsExp = np.exp(self.lastTotals)

      # Sum of all e^totals
      sumTotalsExp = np.sum(totalsExp)

      # Gradients of out[i] against totals
      d_out_d_t = -totalsExp[i] * totalsExp / (sumTotalsExp ** 2)
      d_out_d_t[i] = totalsExp[i] * (sumTotalsExp - totalsExp[i]) / (sumTotalsExp ** 2)

      # Gradients of totals against weights/biases/input
      d_t_d_w = self.lastInputFlattened
      d_t_d_b = 1
      d_t_d_inputs = weights

      # Gradients of loss against totals
      d_L_d_t = gradient * d_out_d_t

      # Gradients of loss against weights/biases/input
      d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
      d_L_d_b = d_L_d_t * d_t_d_b
      d_L_d_inputs = d_t_d_inputs @ d_L_d_t

      # Update weights / biases
      newWeights = weights - learningRate * d_L_d_w
      newBiases = biases - learningRate * d_L_d_b
      weights -= learningRate * d_L_d_w
      biases -= learningRate * d_L_d_b

      return d_L_d_inputs.reshape(self.lastInputShape), newWeights, newBiases
    

## This is a helper class to handle data pre processing
class preProcessData:
  def __init__(
      self, 
      labelsFilePath,
      URLbasePath,
      ImageSaveDirectory
    ) -> None:
    self.labelsFilePath = labelsFilePath
    self.URLbasePath = URLbasePath
    self.ImageSaveDirectory = ImageSaveDirectory

  ## downloads images to local
  def downloadImagesToLocal(self, numberOfImages):
    for index in range(numberOfImages):
      URL = self.URLbasePath + str(index) + ".jpg"
      SAVE_PATH = os.path.join(self.ImageSaveDirectory, str(index) + ".jpeg")
     
      ## Dont Save Image if it already exists
      if os.path.exists(SAVE_PATH):
        continue

      response = requests.get(URL)
      if response.status_code == 200:
        with open(SAVE_PATH, 'wb') as file:
          file.write(response.content)


  ## creates a dictionary of all the labels for each image
  ## {"0.jpeg":1, "2.jpeg":0}
  def createLabelList(self) -> dict:
    labels = pd.read_csv(self.labelsFilePath)
    labelsDict = {}
    for index, row in labels.iterrows():
      labelsDict.update({str(index) + ".jpeg": row['label']})

    return labelsDict
    

## Implementation of CNN Class    
class CNN:
  ## This stores softmax because softmax is only called ONE time
  ## During the first image of the first epoch and never called again
  ## It saves the weights and biases in the class implementation of softmax
  def __init__(self):
    self.softmax: Softmax = None
    self.layers = []


  ## Implementation of adding a layer to CNN
  ## Adds a Conv3x3 Layer followed by a MaxPool2 Layer
  ## To access the Conv3x3 object and the related methods in the layer, 
  ## you can take the first index of the self.layers index
  ## self.layers[0][0] to access the first Conv3x3 Layer
  ## self.layers[1][0] to access the second Conv3x3 Layer
  ## self.layers[0][1] to access the first MaxPool2 Layer

  def addLayer(
    self, 
    conv: Conv3x3,
    pool: MaxPool2
  ):
    self.layers.append([conv, pool])

  ## Returns number of layers
  def numLayers(self):
    return len(self.layers)

  ## Recive as input the label probability 1D matrix and gets the predicted label
  ## Prediction: [.1, .1, .1, .1, .1, .5] So the predicted label would be 6
  ## Then search label dictionary to get correct label
  ## Then compare predicted label with correct label
  ## Returns number of missclassificaitons and cross entropy loss
  def performance(
    self, 
    outputSoftMax,
    correctLabel: int,
    loss: int,
    missclassifications: int,
    ):

    # Calculate cross-entropy loss and missclassifications. np.log() is the natural log.
    loss += -np.log(outputSoftMax[correctLabel])
    predictedClass = np.argmax(outputSoftMax)
    #print("Predicted Class: " + str(predictedClass) + " Correct Class: " + str(correctLabel))
    if predictedClass != correctLabel:
      missclassifications += 1

    return loss, missclassifications
    

  ## Forward Pass of CNN
  ## image is a 2d numpy array
  ## Calls the forward of the Conv3x3 Layer and then calls the forward of the
  ## MaxPool2 Layer
  ## HIGHLY RECOMMENDED TO UNCOMMENT THE PRINT STATEMENTS TO VISUALIZE THE 
  ## DIMENSIONS FROM EACH LAYER
  def forward(
    self, 
    layer, 
    image, 
    ):
    
    # Transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. 
    ## First pass
    if layer == 0:
      
      normalizedImage = (image / 255) - 0.5
      outputConv1 = self.layers[layer][0].forwardPass(normalizedImage, True)
      #print(outputConv1.shape)

      outputPool1 = self.layers[layer][1].forward(outputConv1)
      #print(outputPool1.shape)

      return outputPool1
    
    else:
      normalizedImage = (image / 255) - 0.5
      outputConv = self.layers[layer][0].forwardPass(normalizedImage, False)
      #print(outputConv.shape)

      outputPool = self.layers[layer][1].forward(outputConv)
      #print(outputPool.shape)
      return outputPool
    
  ## Implementation of function to train for one epoch
  ## Initialize the lsos and missclassications to 0. Then 
  ## Then call the forward pass for the CNN
  ## Then get the prediction from softmax
  ## THen update new loss and missclassifications
  ## Then do the forward pass for the CNN
  ## Repeat this for every image in the epoch
  ## HIGHLY RECOMMENDED NOT TO COMMENT OUT THE PRINT STATEMENTS FOR
  ## DEBUGGING PURPOSES AND TO SEE PROGRESS
  def trainEpoch(
    self, 
    learningRate,
    batchSize,
    labelsDictionary,
    weights,
    biases,
    currentEpoch
  ):
    loss = 0
    missclassifications = 0
    for imageNum in range(batchSize):
      imageIndex = str(imageNum) + ".jpeg"
      labelsDictonary = labelsDictionary

      correctLabel = labelsDictonary[imageIndex]
      imagePath = PHOTO_DIRECTORY_PATH + str(imageNum) + ".jpeg"
      image = Image.open(imagePath)
      grayImage = image.convert('L')
      imageArray = np.array(grayImage)

      for layer in range(0, len(self.layers)):
        if layer == 0:

          outputMaxPool = self.forward(layer, imageArray)

        else:
          outputMaxPool = self.forward(layer, outputMaxPool)  



      ## SoftMax instance is only initialized ONCE in the first iteration 
      if imageNum == 0 and currentEpoch == 0:
   
        self.softmax = Softmax(outputMaxPool.shape[1]**2 * outputMaxPool.shape[0] , NUM_LABELS)

      outputSoftMax = self.softmax.forward(outputMaxPool, weights, biases)



      loss, missclassifications = self.performance(outputSoftMax, correctLabel, loss, missclassifications)

      ## Gradient of to be used during backpropogated is intialized here
      ## it is used to optimize the filter and to reconstruct the image
      ## with the feature map
      gradient = np.zeros(NUM_LABELS)
      gradient[correctLabel] = -1 / outputSoftMax[correctLabel]

      #print('DONE WITH FORWARD PASS, BEGINNING BACKWARD PASS\n')

      ##BACKWARD PASS
      gradient, weights, biases = self.softmax.backwardPass(weights, biases, gradient, learningRate)


      ## Go in REVERSE ORDER
      for layer in range(len(self.layers)-1, -1, -1):

        ## If first instance of backward Pass
        if layer == len(self.layers)-1:
          ##Pool First
          gradient = self.layers[layer][1].backward(gradient)
          #Conv3x3 After
          gradient = self.layers[layer][0].backward(gradient, learningRate, False)
        else:

          ##Pool First
          gradient = self.layers[layer][1].backward(gradient)
          #Conv3x3 After
          gradient = self.layers[layer][0].backward(gradient, learningRate, True)

      print("Current Image " + str(imageNum) + ", Current Missclassifications: " + str(missclassifications) + ", Current Loss: " + str(loss))

      
    averageLoss = loss/batchSize
    missclassificationPercentage = missclassifications/batchSize

    print("Current Epoch: " + str(currentEpoch) + ", Missclassification Percentage: " + str(missclassificationPercentage) + ", Average Loss: " + str(averageLoss) + '\n')
      # print("Average Loss: " + averageLoss)
      # print("Missclassification Percentage: " + missclassificationPercentage)

    return weights, biases, averageLoss, missclassificationPercentage

  ## Implementation of function to train CNN for multiple epochs
  ## The weights and biases of softmax to be optimized are 
  ## randomly initalized here. The perofrmance history is also initalized here
  ## after every pass, you add to the performance history
  def train(
    self,
    epochs: int,
    learningRate: int,
    batchSize: int,
    labelsDictionary: dict,
    numberOfLabels: int
  ):
    ## FILTER GRADIENTS (CONV AND POOL LAYERS)

    ## 4 LAYERS = 64 * 7 * 7
    #np.random.seed(SEED_NUMBER)
    weights = np.random.randn((64*7*7), numberOfLabels) / (64*7*7)
    biases = np.zeros(numberOfLabels)

    performanceHistoryDF = pd.DataFrame(columns=["Iteration", "Average Loss", "Missclassification Percentage"])

    for epoch in range(0, epochs):
      weights, biases, averageLoss, missclassificationPercentage =  self.trainEpoch(learningRate, batchSize, labelsDictionary, weights, biases, epoch)

      performanceHistoryDF.loc[epoch] = [epoch, averageLoss, missclassificationPercentage]
      
    return performanceHistoryDF, weights, biases

  ## Prediction of label for single image
  def predictSingleImage(self, imagePath, weights, biases) -> int:
    image = Image.open(imagePath)
    grayImage = image.convert('L')
    imageArray = np.array(grayImage)

    for layer in range(0, len(self.layers)):
        if layer == 0:

          outputMaxPool = self.forward(layer, imageArray)

        else:
          outputMaxPool = self.forward(layer, outputMaxPool)  
          

    prob = self.softmax.forward(outputMaxPool, weights, biases)
    pred = np.argmax(prob)

    return pred

  ## Prediction for a test set
  def predictTestSet(
    self,
    testImages, 
    weights, 
    biases,
    labelDict
    ):
    accuracy = 0
    
    for imageNumber in testImages:
      imagePath = os.path.join(PHOTO_DIRECTORY_PATH, str(imageNumber) +".jpeg")
      predictedLabel = self.predictSingleImage(imagePath, weights, biases)
      actualLabel = labelDict[str(imageNumber)+".jpeg"]
      if predictedLabel == actualLabel:
        accuracy += 1
  
    return accuracy
  
    
## Implementation of a helper class to display for comparing
## Various hyperparameters
class compareHyperParemeters:

  def compareEpochs(self, epochNum1, epochNum2, epochNum3, learningRate, batchSize, labelDict):
    myCNN1 = CNN()
    myCNN1.addLayer(Conv3x3(CONV1_FILTERS), MaxPool2())
    myCNN1.addLayer(Conv3x3(CONV2_FILTERS), MaxPool2())
    myCNN1.addLayer(Conv3x3(CONV3_FILTERS), MaxPool2())
    myCNN1.addLayer(Conv3x3(CONV4_FILTERS), MaxPool2())
    historyOfResults1, weights1, biases1 = myCNN1.train(epochNum1, learningRate, batchSize, labelDict, NUM_LABELS)


    myCNN2 = CNN()
    myCNN2.addLayer(Conv3x3(CONV1_FILTERS), MaxPool2())
    myCNN2.addLayer(Conv3x3(CONV2_FILTERS), MaxPool2())
    myCNN2.addLayer(Conv3x3(CONV3_FILTERS), MaxPool2())
    myCNN2.addLayer(Conv3x3(CONV4_FILTERS), MaxPool2())
    historyOfResults2, weights2, biases2 = myCNN2.train(epochNum2, learningRate, batchSize, labelDict, NUM_LABELS)

    myCNN3 = CNN()
    myCNN3.addLayer(Conv3x3(CONV1_FILTERS), MaxPool2())
    myCNN3.addLayer(Conv3x3(CONV2_FILTERS), MaxPool2())
    myCNN3.addLayer(Conv3x3(CONV3_FILTERS), MaxPool2())
    myCNN3.addLayer(Conv3x3(CONV4_FILTERS), MaxPool2())
    historyOfResults3, weights2, biases2 = myCNN3.train(epochNum3, learningRate, batchSize, labelDict, NUM_LABELS)

    return historyOfResults1, historyOfResults2, historyOfResults3, learningRate, batchSize
  
  ## Shows graph of average loss and missclassifications for different epochs
  def displayCompareEpochs(self, epochNum1, epochNum2, epochNum3, learningRate, batchSize, labelDict):
    historyOfResults1, historyOfResults2, historyOfResults3, learningRate, batchSize = self.compareEpochs(epochNum1, epochNum2, epochNum3, learningRate, batchSize, labelDict)
    ## Average Loss Plot
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(historyOfResults1['Iteration'], historyOfResults1['Average Loss'], label='Epochs: ' + str(epochNum1))
    ax[0].plot(historyOfResults2['Iteration'], historyOfResults2['Average Loss'], label='Epochs: ' + str(epochNum2))
    ax[0].plot(historyOfResults3['Iteration'], historyOfResults3['Average Loss'], label='Epochs: ' + str(epochNum3))
    ax[0].legend()
    ax[0].set(xlabel="Number of Iterations", ylabel="Average Loss")
    ax[0].set_title('Average Loss VS Number of Iterations' + '\n'+ " Batch Size: " + str(batchSize) + ", Learning Rate: "+ str(learningRate))

    ## Number of Missclassifications Plot
    ax[1].plot(historyOfResults1['Iteration'], historyOfResults1['Missclassification Percentage'], label='Epochs: ' + str(epochNum1))
    ax[1].plot(historyOfResults2['Iteration'], historyOfResults2['Missclassification Percentage'], label='Epochs: ' + str(epochNum2))
    ax[1].plot(historyOfResults3['Iteration'], historyOfResults3['Missclassification Percentage'], label='Epochs: ' + str(epochNum3))
    ax[1].legend()
    ax[1].set(xlabel="Number of Iterations", ylabel="Missclassification Percentage")
    ax[1].set_title('Missclassification Percentage VS Number of Iterations' + '\n' + ", Batch Size: " + str(batchSize) + ", Learning Rate: "+ str(learningRate))

    plt.show()

  def compareBatchSizes(self, batchSize1, batchSize2, batchSize3, learningRate, epochs, labelDict):
    myCNN1 = CNN()
    myCNN1.addLayer(Conv3x3(CONV1_FILTERS), MaxPool2())
    myCNN1.addLayer(Conv3x3(CONV2_FILTERS), MaxPool2())
    myCNN1.addLayer(Conv3x3(CONV3_FILTERS), MaxPool2())
    myCNN1.addLayer(Conv3x3(CONV4_FILTERS), MaxPool2())
    historyOfResults1, weights1, biases1 = myCNN1.train(epochs, learningRate, batchSize1, labelDict, NUM_LABELS)


    myCNN2 = CNN()
    myCNN2.addLayer(Conv3x3(CONV1_FILTERS), MaxPool2())
    myCNN2.addLayer(Conv3x3(CONV2_FILTERS), MaxPool2())
    myCNN2.addLayer(Conv3x3(CONV3_FILTERS), MaxPool2())
    myCNN2.addLayer(Conv3x3(CONV4_FILTERS), MaxPool2())
    historyOfResults2, weights2, biases2 = myCNN1.train(epochs, learningRate, batchSize2, labelDict, NUM_LABELS)

    myCNN3 = CNN()
    myCNN3.addLayer(Conv3x3(CONV1_FILTERS), MaxPool2())
    myCNN3.addLayer(Conv3x3(CONV2_FILTERS), MaxPool2())
    myCNN3.addLayer(Conv3x3(CONV3_FILTERS), MaxPool2())
    myCNN3.addLayer(Conv3x3(CONV4_FILTERS), MaxPool2())
    historyOfResults3, weights3, biases3 = myCNN1.train(epochs, learningRate, batchSize3, labelDict, NUM_LABELS)

    return historyOfResults1, historyOfResults2, historyOfResults3, learningRate, epochs
  

  ## Shows graph of average loss and missclassifications for different batch sizes
  def displayCompareBatchSize(self, batchSize1, batchSize2, batchSize3, learningRate, epochs, labelDict):
    historyOfResults1, historyOfResults2, historyOfResults3, learningRate, epochs = self.compareBatchSizes(batchSize1, batchSize2, batchSize3, learningRate, epochs, labelDict)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax = list(ax) 

    ## Average Loss Plot
    ax[0].plot(historyOfResults1['Iteration'], historyOfResults1['Average Loss'], label='Batch Size: ' + str(batchSize1))
    ax[0].plot(historyOfResults2['Iteration'], historyOfResults2['Average Loss'], label='Batch Size: ' + str(batchSize2))
    ax[0].plot(historyOfResults3['Iteration'], historyOfResults3['Average Loss'], label='Batch Size: ' + str(batchSize3))
    ax[0].legend()
    ax[0].set(xlabel="Number of Iterations", ylabel="Average Loss")
    ax[0].set_title('Average Loss VS Number of Iterations' + '\n'+ " Epochs: " + str(epochs) + ", Learning Rate: "+ str(learningRate))

    ## Number of Missclassifications Plot
    ax[1].plot(historyOfResults1['Iteration'], historyOfResults1['Missclassification Percentage'], label='Batch Size: ' + str(batchSize1))
    ax[1].plot(historyOfResults2['Iteration'], historyOfResults2['Missclassification Percentage'], label='Batch Size: ' + str(batchSize2))
    ax[1].plot(historyOfResults3['Iteration'], historyOfResults3['Missclassification Percentage'], label='Batch Size: ' + str(batchSize3))
    ax[1].legend()
    ax[1].set(xlabel="Number of Iterations", ylabel="Missclassification Percentage")
    ax[1].set_title('Missclassification Percentage VS Number of Iterations' + '\n' + ", Epochs: " + str(epochs) + ", Learning Rate: "+ str(learningRate))

    plt.show()

    return historyOfResults1.loc[epochs-1]['Average Loss'], historyOfResults1.loc[epochs-1]['Missclassification Percentage'], historyOfResults2.loc[epochs-1]['Average Loss'], historyOfResults2.loc[epochs-1]['Average Loss'], historyOfResults3.loc[epochs-1]['Average Loss'], historyOfResults3.loc[epochs-1]['Average Loss']


  def compareLearningRates(self, learningRate1, learningRate2, learningRate3, batchSize, epochs, labelDict):

    myCNN1 = CNN()
    myCNN1.addLayer(Conv3x3(CONV1_FILTERS), MaxPool2())
    myCNN1.addLayer(Conv3x3(CONV2_FILTERS), MaxPool2())
    myCNN1.addLayer(Conv3x3(CONV3_FILTERS), MaxPool2())
    myCNN1.addLayer(Conv3x3(CONV4_FILTERS), MaxPool2())
    historyOfResults1, weights1, biases1 = myCNN1.train(epochs, learningRate1, batchSize, labelDict, NUM_LABELS)


    myCNN2 = CNN()
    myCNN2.addLayer(Conv3x3(CONV1_FILTERS), MaxPool2())
    myCNN2.addLayer(Conv3x3(CONV2_FILTERS), MaxPool2())
    myCNN2.addLayer(Conv3x3(CONV3_FILTERS), MaxPool2())
    myCNN2.addLayer(Conv3x3(CONV4_FILTERS), MaxPool2())
    historyOfResults2, weights2, biases2 = myCNN1.train(epochs, learningRate2, batchSize, labelDict, NUM_LABELS)

    myCNN3 = CNN()
    myCNN3.addLayer(Conv3x3(CONV1_FILTERS), MaxPool2())
    myCNN3.addLayer(Conv3x3(CONV2_FILTERS), MaxPool2())
    myCNN3.addLayer(Conv3x3(CONV3_FILTERS), MaxPool2())
    myCNN3.addLayer(Conv3x3(CONV4_FILTERS), MaxPool2())
    historyOfResults3, weights3, biases3 = myCNN1.train(epochs, learningRate3, batchSize, labelDict, NUM_LABELS)

    return historyOfResults1, historyOfResults2, historyOfResults3, batchSize, epochs
  
  ## Shows graph of average loss and missclassifications for different learning rates
  def displayCompareLearningRates(self, learningRate1, learningRate2, learningRate3, batchSize, epochs, labelDict):
    historyOfResults1, historyOfResults2, historyOfResults3, batchSize, epochs = self.compareLearningRates(learningRate1, learningRate2, learningRate3, batchSize, epochs, labelDict)
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ## Average Loss Plot
    ax[0].plot(historyOfResults1['Iteration'], historyOfResults1['Average Loss'], label='Learning Rate: ' + str(learningRate1))
    ax[0].plot(historyOfResults2['Iteration'], historyOfResults2['Average Loss'], label='Learning Rate: ' + str(learningRate2))
    ax[0].plot(historyOfResults3['Iteration'], historyOfResults3['Average Loss'], label='Learning Rate: ' + str(learningRate3))
    ax[0].legend()
    ax[0].set(xlabel="Number of Iterations", ylabel="Average Loss")
    ax[0].set_title('Average Loss VS Number of Iterations' + '\n'+ " Epochs: " + str(epochs) + ", Batch Size: "+ str(batchSize))

    ## Number of Missclassifications Plot
    ax[1].plot(historyOfResults1['Iteration'], historyOfResults1['Missclassification Percentage'], label='Learning Rate ' + str(learningRate1))
    ax[1].plot(historyOfResults2['Iteration'], historyOfResults1['Missclassification Percentage'], label='Learning Rate ' + str(learningRate2))
    ax[1].plot(historyOfResults3['Iteration'], historyOfResults1['Missclassification Percentage'], label='Learning Rate ' + str(learningRate3))
    ax[1].legend()
    ax[1].set(xlabel="Number of Iterations", ylabel="Missclassification Percentage")
    ax[1].set_title('Missclassification Percentage VS Number of Iterations' + '\n' + ", Epochs: " + str(epochs) + ", Batch Size: "+ str(batchSize))

    plt.show()
    
    return historyOfResults1.loc[epochs-1]['Average Loss'], historyOfResults1.loc[epochs-1]['Missclassification Percentage'], historyOfResults2.loc[epochs-1]['Average Loss'], historyOfResults2.loc[epochs-1]['Average Loss'], historyOfResults3.loc[epochs-1]['Average Loss'], historyOfResults3.loc[epochs-1]['Average Loss']

  ## This does not compare anything but just shows the training history
  ## for the given hyperparameters
  ## Use this if you don't want to compare anything and just want to ssee the training history
  def displayTrainingHistory(self, EPOCHS, LEARNING_RATE, BATCH_SIZE, labelDict, NUM_LABELS):
    myCNN = CNN()
    myCNN.addLayer(Conv3x3(CONV1_FILTERS), MaxPool2())
    myCNN.addLayer(Conv3x3(CONV2_FILTERS), MaxPool2())
    myCNN.addLayer(Conv3x3(CONV3_FILTERS), MaxPool2())
    myCNN.addLayer(Conv3x3(CONV4_FILTERS), MaxPool2())

    performanceHistoryDF, weights, biases = myCNN.train(EPOCHS, LEARNING_RATE, BATCH_SIZE, labelDict, NUM_LABELS)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(performanceHistoryDF['Iteration'], performanceHistoryDF['Average Loss'])
    ax[0].set(xlabel="Number of Iterations", ylabel="Average Loss")
    ax[0].set_title('Average Loss VS Number of Iterations for Training' + '\n' + " Epochs: " + str(EPOCHS) + ", Batch Size: " + str(BATCH_SIZE) + ", Learning Rates: " + str(LEARNING_RATE) + '\n' + " Final Average Loss: " + str(performanceHistoryDF.loc[EPOCHS-1]['Average Loss']))

    ax[1].plot(performanceHistoryDF['Iteration'], performanceHistoryDF['Missclassification Percentage'])
    ax[1].set(xlabel="Number of Iterations", ylabel="Missclassification Percentage")
    ax[1].set_title('Missclassification Percentage VS Number of Iterations for Training' + '\n' + " Epochs: " + str(EPOCHS) + ", Batch Size: " + str(BATCH_SIZE) + ", Learning Rates: " + str(LEARNING_RATE) + '\n' + " Final Missclassification Percentage: " + str(performanceHistoryDF.loc[EPOCHS-1]['Missclassification Percentage']))


    plt.show()

    return performanceHistoryDF.loc[EPOCHS-1]['Average'], performanceHistoryDF.loc[EPOCHS-1]['Missclassification Percentage']


  ## Single prediction helper function
  def trainAndGetSinglePrediction(self, EPOCHS, LEARNING_RATE, BATCH_SIZE, labelDict, NUM_LABELS, imageNumber):
    IMAGE = os.path.join(PHOTO_DIRECTORY_PATH, str(imageNumber) +".jpeg")
    myCNN = CNN()
    myCNN.addLayer(Conv3x3(CONV1_FILTERS), MaxPool2())
    myCNN.addLayer(Conv3x3(CONV2_FILTERS), MaxPool2())
    myCNN.addLayer(Conv3x3(CONV3_FILTERS), MaxPool2())
    myCNN.addLayer(Conv3x3(CONV4_FILTERS), MaxPool2())

    performanceHistoryDF, weights, biases = myCNN.train(EPOCHS, LEARNING_RATE, BATCH_SIZE, labelDict, NUM_LABELS)
    predictionLabel = myCNN.predictSingleImage(IMAGE, weights, biases)

    print("Actual Label: " + str(labelDict[str(imageNumber)+".jpeg"]) + ", Predicted Label: " + str(predictionLabel) + '\n' + "Average Loss: " + str(performanceHistoryDF.loc[EPOCHS-1]['Average Loss']) + '\n' + "Missclassification Percentage: " + str(performanceHistoryDF.loc[EPOCHS-1]['Missclassification Percentage']))

  ## Predict on test dataset
  def predictOnTestSet(self, EPOCHS, LEARNING_RATE, BATCH_SIZE, labelDict, NUM_LABELS, sizeOfTestSet):
    dataSet = [i for i in range(NUM_IMAGES_TO_DOWNLOAD)]
    random_indices = random.sample(range(len(dataSet)), sizeOfTestSet)

    testSet = [dataSet[index] for index in random_indices]
    myCNN = CNN()
    myCNN.addLayer(Conv3x3(CONV1_FILTERS), MaxPool2())
    myCNN.addLayer(Conv3x3(CONV2_FILTERS), MaxPool2())
    myCNN.addLayer(Conv3x3(CONV3_FILTERS), MaxPool2())
    myCNN.addLayer(Conv3x3(CONV4_FILTERS), MaxPool2())

    performanceHistoryDF, weights, biases = myCNN.train(EPOCHS, LEARNING_RATE, BATCH_SIZE, labelDict, NUM_LABELS)

    correctPredictions = myCNN.predictTestSet(testSet, weights, biases, labelDict)
    accuracyPercent = correctPredictions/sizeOfTestSet
    print("Accuracy on Test Set: " + str(accuracyPercent) + ", with Learning Rate: " + LEARNING_RATE + ", Epochs: " + EPOCHS + ", Batch Size: " + BATCH_SIZE)


## Change these environment variables to run the code ##
## So you need to have a directory for photos and download the train.csv.
## Change the PHOTO_DIRECTORY_PATH and labelsCSV variable to YOUR LOCAL variable

PHOTO_DIRECTORY_PATH = r"/Users/pham/Desktop/School/Summer 2023/CS 6375/Project/Photos/"
labelsCSV = r"/Users/pham/Desktop/School/Summer 2023/CS 6375/Project copy/train.csv"
URL_BASE_PATH = "https://personal.utdallas.edu/~kml190002/project-cs6375/train/"

#### Hyper Parameters ####
## DONT CHANGE THESE 
NUM_LABELS = 6
SEED_NUMBER = 2

## Recommended default parameters (Dont change these)
DEFAULT_BATCH_SIZE = 50
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.0001

## Number of Filters for each Convolution Layer (Dont chnage these too)
CONV1_FILTERS = 8
CONV2_FILTERS = 16
CONV3_FILTERS = 32
CONV4_FILTERS = 64

#### CHANGE THESE FOR THE COMPARISON CLASS ####

## PREDICT IMAGE
NUM_IMAGES_TO_DOWNLOAD = 100
IMAGE_NUM = 1

## COMPARE EPOCHS
EPOCH_1 = 10
EPOCH_2 = 20
EPOCH_3 = 30

## COMPARE BATCH SIZE
BATCH_SIZE_1 = 50
BATCH_SIZE_2 = 100
BATCH_SIZE_3 = 200

## COMPARE LEARNING RATES
LEARNING_RATE_1 = 0.0001
LEARNING_RATE_2 = 0.0005
LEARNING_RATE_3 = 0.00001


## PREDICTION ON TEST
SIZE_OF_TEST_SET = 20


## Main Function
def main():
  
  ## This is sees how long it takes
  startTime = time.time()

  ## This creates the labelDictionary to be used
  myData = preProcessData(labelsCSV, URL_BASE_PATH, PHOTO_DIRECTORY_PATH)
  myData.downloadImagesToLocal(NUM_IMAGES_TO_DOWNLOAD)
  labelDict = myData.createLabelList()
  

  ## This is just an example 
  # myCNN = CNN()
  # myCNN.addLayer(Conv3x3(CONV1_FILTERS), MaxPool2())
  # myCNN.addLayer(Conv3x3(CONV2_FILTERS), MaxPool2())
  # myCNN.addLayer(Conv3x3(CONV3_FILTERS), MaxPool2())
  # myCNN.addLayer(Conv3x3(CONV4_FILTERS), MaxPool2())

  # performanceHistoryDF, weights, biases = myCNN.train(EPOCHS, LEARNING_RATE, BATCH_SIZE, labelDict, NUM_LABELS)
  # predictionLabel = myCNN.predict(r"/Users/pham/Desktop/School/Summer 2023/CS 6375/Project/Photos/49.jpeg", weights, biases)
  # print(labelDict['49.jpeg'])
  # print(predictionLabel)



  ## This creates the object to compare the neural networks
  ## Uncomment the function that you need
  compare = compareHyperParemeters()

  ## Show the training history ##
  #compare.displayTrainingHistory(100, 0.0001, 1000, labelDict, NUM_LABELS)
  #compare.displayTrainingHistory(DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE, labelDict, NUM_LABELS)
  
  ## Predict Single Image ## 
  #compare.trainAndGetLabelPredictionOfImage(DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE, labelDict, NUM_LABELS, IMAGE_NUM)

  ## Compare Different Epochs ##
  #compare.displayCompareEpochs(EPOCH_1, EPOCH_2, EPOCH_3, DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE, labelDict)
  
  ## Compare Different Batch Sizes ##
  #compare.displayCompareBatchSize(BATCH_SIZE_1, BATCH_SIZE_2, BATCH_SIZE_3, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, labelDict)

  ## Compare Different Learning Rates ##
  #compare.displayCompareLearningRates(LEARNING_RATE_1, LEARNING_RATE_2, LEARNING_RATE_3, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, labelDict)

  ## Predict on Test Set ##
  #compare.predictOnTestSet(DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE, labelDict, NUM_LABELS, 20)
  
  
  endTime = time.time()
  print(startTime - endTime)
if __name__ == "__main__":
  main()