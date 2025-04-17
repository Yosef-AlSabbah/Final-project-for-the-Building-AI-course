<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Smart Recycling Advisor

Final project for the Building AI course

## Summary

Smart Recycling Advisor leverages computer vision and deep learning to optimize recycling workflows by automatically identifying and sorting waste. Building AI course project.

## Background

Recycling facilities often face inefficiencies due to manual sorting which increases costs, error rates, and delays. Smart Recycling Advisor aims to automate this process, enhancing purity and reducing waste management costs.  
* Manual waste sorting is labor-intensive and error-prone  
* Automation can lead to improved recycling purity and environmental benefits  
* Personal motivation: I am passionate about environmental sustainability and believe that AI can create significant social impact.

## How is it used?

The system is designed to be deployed at recycling facilities where cameras capture real-time footage of waste streams. The AI model processes these images to classify waste types (plastic, metal, paper, etc.). Operators can monitor the system through a web dashboard that shows real-time statistics and alerts.  
- **Users:** Facility managers and recycling plant operators  
- **Environment:** Industrial sorting facilities, often with varying lighting and operational conditions  
- **Process:** Visual data is captured, processed in near-real-time, and used to control sorting mechanisms automatically.

![Smart Recycling Advisor](https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg)

<img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg" width="300">

## Data sources and AI methods

**Data Sources:**
- Video streams from high-resolution cameras installed along waste sorting lines.
- Historical image datasets from previous recycling studies.
- Sensor data from the sorting mechanism.

**AI Methods:**
- Convolutional Neural Networks (CNNs) for image classification.
- Transfer Learning to leverage pre-trained models.
- Real-time stream processing and anomaly detection.

Example code snippet for a simple CNN model using TensorFlow:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Example usage:
input_shape = (128, 128, 3)  # 128x128 pixel images with 3 color channels
num_classes = 4              # classes: plastic, metal, paper, waste
model = create_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## Challenges

While Smart Recycling Advisor promises efficiency improvements, several challenges remain:
- **Data Variability:** Image quality might vary with lighting conditions and camera angles.
- **Edge Cases:** Some waste items may not clearly belong to a single category.
- **Deployment:** Integrating the system into legacy plant operations may require significant retrofitting.
- **Ethics:** Ensuring data privacy and secure handling of facility data is crucial.

## What next?

Future improvements for Smart Recycling Advisor could include:
- Expansion of the AI model to handle a wider range of waste categories.
- Integration with IoT devices for more precise sorting control.
- Adding predictive maintenance features for recycling machinery.
- Collaborations with waste management companies for large-scale trials.
- Enhancement of the live dashboard with detailed analytics and AI-driven insights.

## Acknowledgments

- Many thanks to the Building AI course by Reaktor Innovations and the University of Helsinki for inspiring this project.
- Gratitude to open source communities for providing tools like TensorFlow, which made this project possible.
- Special thanks to industry partners and municipal recycling initiatives that continue to drive innovation in sustainable practices.

For further information or collaboration opportunities, please reach out via the project's GitHub repository.

---

*Building AI course project*
