# CIFAR-10 Image Classification
Kaggle notebook for classifying images from the CIFAR-10 dataset into their corresponding classes using Convolutional Neural Networks (CNN) and transfer learning techniques.

![App Screenshot](https://paritkansal121.odoo.com/web/image/341-e671e2ce/dataset-cover%20%281%29.webp)

### Kaggle Notebook:
Link: https://www.kaggle.com/code/paritkansal/cifar-10

### Project Overview
Image classification is a fundamental task in computer vision, and the CIFAR-10 dataset is a popular benchmark for evaluating classification models. In this blog, we’ll explore the process of classifying images from the CIFAR-10 dataset, covering the dataset overview, data preprocessing, model building, and performance evaluation.

### CIFAR-10 Dataset Overview
The CIFAR-10 dataset consists of 60,000 color images, each with a resolution of 32x32 pixels. These images are categorized into 10 different classes, with each class containing 6,000 images. 

The classes are:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is divided into 50,000 training images and 10,000 test images, making it a valuable resource for benchmarking image classification models.

### Data Preprocessing
Before feeding the images into a model, preprocessing is essential to ensure optimal performance. The preprocessing steps typically include:

**Normalization**: Scaling pixel values to a range between 0 and 1. This helps the model learn more effectively by ensuring that input values are in a consistent range.

**Resizing**: While CIFAR-10 images are originally 32x32 pixels, resizing them to larger dimensions (such as 96x96 pixels) can help capture more detailed features. This resizing step is crucial for improving model accuracy.

### Model Building and Training
We explored two primary approaches for image classification on the CIFAR-10 dataset:

**Convolutional Neural Networks (CNNs)**: CNNs are designed specifically for image data. They use convolutional layers to automatically extract features from images, followed by pooling layers and fully connected layers to classify the images. A typical CNN involves several convolutional and pooling layers followed by dense layers to make predictions.

**Transfer Learning with InceptionV3**: Transfer Learning leverages pre-trained models to improve performance on new tasks. InceptionV3, a model pre-trained on a large dataset like ImageNet, was used as a base. We added custom layers on top of this pre-trained model to fine-tune it for CIFAR-10 classification. This approach allows the model to benefit from the features learned by the pre-trained model, often leading to better performance.

### Evaluation and Results
After training both models, their performance was evaluated using the test set to measure accuracy and loss:

- **CNN Model**: This model was evaluated based on how well it classified the test images into the correct categories. The CNN model achieved a test accuracy of 70.12%, demonstrating its capability to classify images with reasonable effectiveness.

- **InceptionV3 Transfer Learning Model**: Transfer Learning models typically outperform models built from scratch. By using a pre-trained model as a base, the Transfer Learning approach achieved a test accuracy of 82.33%. This superior performance highlights the benefits of leveraging pre-trained models for image classification tasks.

### Results Comparison
In the results comparison, the Transfer Learning model with InceptionV3 showed superior performance compared to the CNN model built from scratch. The higher accuracy and lower loss of the InceptionV3 model demonstrate the advantages of using pre-trained models. Transfer Learning leverages existing knowledge from large datasets, leading to improved classification accuracy and reduced loss.

### Conclusion
The CIFAR-10 dataset serves as an excellent benchmark for image classification tasks. By employing both CNNs and Transfer Learning, we demonstrated effective methods for building and evaluating image classification models. While CNNs offer a solid foundation for image classification, Transfer Learning with pre-trained models like InceptionV3 often provides enhanced performance, making it a powerful tool for tackling complex classification problems.

Experimenting with different model architectures and hyperparameters can further optimize performance. Whether using CNNs or Transfer Learning, understanding the nuances of the CIFAR-10 dataset and preprocessing steps is key to achieving high classification accuracy.

