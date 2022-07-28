# Eye.AI
### The Code Repository For Eye.AI <br>
<br>

## Application Demo & Elevator Pitch
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/H3gy18s9K-M/maxresdefault.jpg)](https://www.youtube.com/watch?v=H3gy18s9K-M)

## APIs
- ### [Blindness Detection (Diabetic Retinopathy) : 94% Accuracy](https://eyeai-blindness.herokuapp.com/)
- ### [Eye Disease Detection (Cataracts, Glaucoma & Uveitis) : 99% Accuracy](https://eyeai-eyedisease.herokuapp.com/)
- ### [Corneal Ulcers (Point-Like, Flaky & Healthy) : 97% Accuracy](https://eyeai-cornealulcers.herokuapp.com/)
`Note: These APIs can be tested out through PostMan. The POST request has been used to receive images and predictions.`
<br>

## Contents
- ### Code
  - #### [Eye Disease](https://github.com/Ansh3101/Eye.AI/tree/main/Code/Cataract%2C%20Glaucoma%20%26%20Uveitis/)
  - #### [Corneal Ulcers](https://github.com/Ansh3101/Eye.AI/tree/main/Code/Corneal%20Ulcers/)
  - #### [Blindness Detection (Diabetic Retinopathy)](https://github.com/Ansh3101/Eye.AI/tree/main/Code/Blindness%20Detection/)
- ### Application
  - #### [APIs](https://github.com/Ansh3101/Eye.AI/blob/main/Application/Post.cs)
  - #### [Landing Page](https://github.com/Ansh3101/Eye.AI/blob/main/Application/MainWindow.xaml)
  - #### [Disease Election](https://github.com/Ansh3101/Eye.AI/blob/main/Application/SelectionPage.xaml)
  - #### [Image Upload](https://github.com/Ansh3101/Eye.AI/blob/main/Application/PredictionPage.xaml)
  - #### [Predictions](https://github.com/Ansh3101/Eye.AI/blob/main/Application/ShowPrediction.xaml)
- ### APIs (Code)
  - #### [Eye Disease](https://github.com/Ansh3101/Eye.AI/tree/main/APIs/Eye%20Disease/)
  - #### [Corneal Ulcers](https://github.com/Ansh3101/Eye.AI/tree/main/APIs/Corneal%20Ulcers/)
  - #### [Blindness Detection](https://github.com/Ansh3101/Eye.AI/tree/main/APIs/Blindness/)

## Code

### Cataracts, Glaucoma And Uveitis
#### [The Dataset](https://github.com/Ansh3101/Eye.AI/tree/main/Code/Cataract%2C%20Glaucoma%20%26%20Uveitis/Data/)
The dataset contains high-quality, well augmented images of eyes diagnosed with cataracts, glaucoma, uveitis and a control class of healthy images.
#### [Data Augmentation](https://github.com/Ansh3101/Eye.AI/blob/main/Code/Cataract%2C%20Glaucoma%20%26%20Uveitis/DataAugmentation.ipynb)
The code notebook that augmented images so as to increase the number of images in the dataset with augmentations.
#### [Training](https://github.com/Ansh3101/Eye.AI/blob/main/Code/Cataract%2C%20Glaucoma%20%26%20Uveitis/Training.ipynb)
The code notebook used to train the EfficientNetV2 Model which was used after much experimentation with the ResNexT-101 Architecture. Our model managed to achieve 99% accuracy on the validation dataset after hyperparameter tuning!
#### [Prediction](https://github.com/Ansh3101/Eye.AI/blob/main/Code/Cataract%2C%20Glaucoma%20%26%20Uveitis/Prediction.ipynb)
The test notebook used to generate predictions on single images. This same code was used in the API.
#### [Model Weights](https://github.com/Ansh3101/Eye.AI/blob/main/Code/Cataract%2C%20Glaucoma%20%26%20Uveitis/EyeDisease.pth)
The weights of the model that managed to achieve 99% accuracy on the test set.
<br><br>

### Diabetic Retinopathy
#### [The Dataset](https://github.com/Ansh3101/Eye.AI/tree/main/Code/Blindness%20Detection/Data%20(Test)/)
The dataset contains high-quality, well augmented images of corneal x-rays of the eye diagnosed with different stages of DR as well as a control class of healthy corneal x-rays.
#### [Data Augmentation](https://github.com/Ansh3101/Eye.AI/blob/main/Code/Blindness%20Detection/DataAugmentation.ipynb)
The code notebook that augmented images so as to increase the number of images in the dataset with augmentations.
#### [Training](https://github.com/Ansh3101/Eye.AI/blob/main/Code/Blindness%20Detection/Training.ipynb)
The code notebook used to train the original EfficientNet Model which was used after much experimentation with the InceptionNetV3 Architecture. Our model managed to achieve 94% accuracy on the validation dataset after hyperparameter tuning!
#### [Prediction](https://github.com/Ansh3101/Eye.AI/blob/main/Code/Blindness%20Detection/Prediction.ipynb)
The test notebook used to generate predictions on single images. This same code was used in the API.
#### [Model Weights](https://github.com/Ansh3101/Eye.AI/blob/main/Code/Blindness%20Detection/blindness.pth)
The weights of the model that managed to achieve 94% accuracy on the test set.
<br><br>

### Corneal Ulcers
#### [The Dataset](https://github.com/Ansh3101/Eye.AI/tree/main/Code/Corneal%20Ulcers/Data/)
The dataset contains high-quality, well augmented images of eyes diagnosed with point corneal ulcers, flaky corneal ulcers, uveitis and a control class of healthy images of eyes doused with flourescein powder under UV.
#### [Data Augmentation](https://github.com/Ansh3101/Eye.AI/blob/main/Code/Corneal%20Ulcers/DataAugmentation.ipynb)
The code notebook that augmented images so as to increase the number of images in the dataset with augmentations.
#### [Training](https://github.com/Ansh3101/Eye.AI/blob/main/Code/Corneal%20Ulcers/Training.ipynb)
The code notebook used to train the EfficientNetV2 Model which was used after much experimentation with the ResNet-180 Architecture. Our model managed to achieve 96% accuracy on the validation dataset after hyperparameter tuning!
#### [Prediction](https://github.com/Ansh3101/Eye.AI/blob/main/Code/Corneal%20Ulcers/Prediction.ipynb)
The test notebook used to generate predictions on single images. This same code was used in the API.
#### [Model Weights](https://github.com/Ansh3101/Eye.AI/blob/main/Code/Corneal%20Ulcers/cornealulcers.pth)
The weights of the model that managed to achieve 96% accuracy on the test set.
<br><br><br>

## Application
<br>

## APIs
<br>

<!--
## APIs

### Public Links (Use POST Method On Postman)
- ### [Blindness Detection (Diabetic Retinopathy) : 94% Accuracy](https://eyeai-blindness.herokuapp.com/)
- ### [Eye Disease Detection (Cataracts, Glaucoma & Uveitis) : 98% Accuracy](https://eyeai-eyedisease.herokuapp.com/)
- ### [Corneal Ulcers (Point-Like, Flaky & Healthy) : 96% Accuracy](https://eyeai-cornealulcers.herokuapp.com/)
<br><br>

### [Elevator Pitch And Code Walkthrough](https://youtu.be/H3gy18s9K-M)
-->
