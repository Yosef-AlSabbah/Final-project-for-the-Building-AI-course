
# Smart Recycling Advisor

Final project for the Building AI course

## Summary

Smart Recycling Advisor leverages computer vision and deep learning to optimize recycling workflows by automatically identifying and sorting different types of waste. Building AI course project.

## Background

Recycling facilities often face inefficiencies because manual sorting increases labor costs and error rates. Smart Recycling Advisor aims to automate waste classification to enhance recycling purity and reduce operational costs.
- Manual waste sorting is labor-intensive and error-prone.
- Automation can significantly improve recycling efficiency and sustainability.
- Personal motivation: Passion for environmental sustainability and making a positive impact on waste management.

## How is it used?

The system is designed to be deployed in recycling facilities:
- **Data Acquisition:** Cameras capture real-time video streams of waste on a conveyor belt.
- **Processing:** The AI model processes the images to classify waste into categories such as plastic, metal, paper, or glass.
- **Dashboard:** Results are displayed on a web dashboard for facility operators to monitor performance and handle exceptions.

![Smart Recycling Advisor](https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg)

<img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg" width="300">

## Data Sources and AI Methods

**Data Sources:**
- Real-time video streams from cameras at recycling facilities.
- Historical image datasets collected from previous recycling studies.
- Sensor data from sorting machinery.

**AI Methods:**
- Convolutional Neural Networks (CNNs) for image classification.
- Transfer Learning using pre-trained models (e.g., MobileNet, ResNet).
- Real-time processing to classify waste types as images are captured.

_Example Code for a basic CNN model:_
```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = (128, 128, 3)
num_classes = 4
model = create_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## Challenges

- **Data Variability:** Variations in lighting, camera angle, and environmental conditions.
- **Edge Cases:** Some waste items might not clearly belong to one category.
- **Integration:** Adapting the solution to different facility setups may require customization.
- **Ethical Considerations:** Ensuring data privacy and transparency in automated decision-making.

## What Next?

Future improvements could include:
- Expanding the AI model to handle more waste categories.
- Integrating IoT sensors for enhanced monitoring.
- Developing a robust web dashboard for real-time analytics and maintenance alerts.
- Collaborating with recycling companies for pilot testing and iterative improvements.

## Acknowledgments

- Thanks to the Building AI course by Reaktor Innovations and the University of Helsinki for inspiring this project.
- Gratitude to the open-source communities for tools such as TensorFlow, Keras, and OpenCV.
- Special thanks to industry partners driving sustainable practices in waste management.

*Building AI course project*
