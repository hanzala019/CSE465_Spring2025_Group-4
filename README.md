# CSE465_Spring2025_Group-4
Project for CSE465 on 'Nurse Care Activity Recognition' using Machine Learning.
## Team Contributions

| Name                                      | Email                              | Contact No.   | Contribution         |
|-------------------------------------------|------------------------------------|--------------|----------------------|
| Abdullah Mohammad Muntasir Adnan Jami     | abdullah.jami@northsouth.edu      | 1612109658   | [Senior Developer] (40%) |
| Hasan Bin Omar                            | hasan.omar@northsouth.edu         | 1301769242   | [Junior Developer] (30%) |
| Abrar Ur Alam                             | abrar.alam@northsouth.edu         | 1733387871   | [Junior Developer] (30%)|

Everyone mostly had a hand in all the parts starting with dataset cleaning, combining and training a model.
Hasan and abrar did majority of the dataset cleaning part and Jami trained the model.
Our senior developer, Adnan bhai, worked harder than the rest of us and guided us through out the whole process.

## Architecture

This model is built to classify different activities from short sequences of accelerometer data.
It starts with an LSTM layer that takes in 15 inputs, 5 each of  x, y, z axis values and outputs a 64-dimensional vector.
This vector is then passed to a hidden layer with 32 units and ReLU activation that helps pick out important features.
Finally, the model uses a softmax layer to guess any of the 4 activity the data represents by giving a score to each possible activity.




## Test result
| Metric                | Average Accuracy | Average Precision | Average Recall | Average F1-Score |
|-----------------------|------------------|-------------------|----------------|------------------|
| Value                 | 74.52            | 76.17             | 74.52          | 74.24            |

![Network Diagram](Final%20Project/confusion.jpg)

## Dataset Link
[Download from Google Drive](https://drive.google.com/file/d/1PLSxD0UMmuWyphyazXXFiAq-FMOx5nJV/view?usp=sharing)
