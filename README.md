# CSE465_Spring2025_Group-4
Project for CSE465 on 'Nurse Care Activity Recognition' using Machine Learning.
## Team Contributions

| Name                                      | Email                              | Contact No.   | Contribution         |
|-------------------------------------------|------------------------------------|--------------|----------------------|
| Abdullah Mohammad Muntasir Adnan Jami     | abdullah.jami@northsouth.edu      | 1612109658   | [Senior Developer] |
| Hasan Bin Omar                            | hasan.omar@northsouth.edu         | 1301769242   | [Junior Developer] |
| Abrar Ur Alam                             | abrar.alam@northsouth.edu         | 1733387871   | [Junior Developer] |

## Data Augmentation Technique

The dataset is augmented using **noise addition, and scaling** to increase variability and improve model generalization.

### Steps:

1. **Adding Noise**:  
   - Gaussian noise (**mean = 0**, **std = 0.10**) is added to sensor readings to simulate real-world measurement variations.

2. **Scaling**:  
   - Each sensor value is multiplied by a random scaling factor in the range **(0.94, 1.06)** to introduce slight variations in magnitude.

### Benefits:
- Improves model robustness to sensor variability.
- Helps prevent overfitting by introducing slight variations.
- Expands the dataset without requiring additional data collection.
## Five fold cross validation result
| Metric       | Value |
|------------------|-----------|
| Average Accuracy | 0.7903    |
| Average Precision| 0.7767    |
| Average Recall   | 0.7903    |
| Average F1-Score | 0.7811    |
