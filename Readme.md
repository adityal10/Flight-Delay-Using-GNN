
# Flight Delay Prediction using Graph Neural Networks (GNN)

## Project Overview

This project aims to predict flight delays using a Graph Neural Network (GNN) model, which leverages both the relational structure of flight routes and the features of individual flights. The model was trained and evaluated using a dataset containing flight information, with each flight treated as a node and flight connections treated as edges. The approach aims to provide an efficient and effective way to predict whether a flight will be delayed based on its features and its relationships with other flights.

---

## Table of Contents
1. [Business Statement](#business-statement)
2. [Dataset](#dataset)
3. [Modeling Approach](#modeling-approach)
4. [GNN Model Architecture](#gnn-model-architecture)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [Installation and Usage](#installation-and-usage)
8. [Files and Structure](#files-and-structure)
9. [License](#license)

---

## Business Statement

Flight delays are a common issue in air travel, causing inconvenience for passengers and additional costs for airlines. By predicting flight delays in advance, airlines can optimize their schedules, improve customer service, and minimize operational disruptions. This project uses a GNN-based approach, which leverages both flight-specific features (like departure and arrival times, delay history, etc.) and the network structure of the flights themselves.

### Why GNN?
A GNN model was chosen over traditional machine learning models because:
- **Relational Structure:** Flights are interconnected through common routes and airports, forming a natural graph structure. Traditional machine learning models fail to capture these relationships effectively.
- **Network Learning:** GNNs excel in capturing node relationships and propagate information across the graph, making them suitable for predicting flight delays based on interdependencies.
- **Drawback:** GNNs can be more complex and computationally expensive compared to traditional models. However, for large-scale networked data (such as flight schedules), GNNs provide a significant performance advantage.

---

## Dataset

The dataset used for this project contains detailed flight records, including both node-specific features (individual flight details) and edge connections (relationships between flights). The columns include:

- **Flight Information:** `FL_DATE`, `AIRLINE`, `FL_NUMBER`, `ORIGIN`, `DEST`, `DEP_TIME`, `ARR_TIME`, `DELAY_TYPE` (delayed or on-time), etc.
- **Edge Information:** Flights are connected based on shared airports and temporal proximity.

### Preprocessing:
- **Node Features:** Flights were encoded with attributes like airline, origin, destination, departure time, etc.
- **Edge Creation:** Edges were constructed based on flights sharing the same origin or destination airport and occurring close to each other in time.
- **Train-Test Split:** A 20% test set was created, and 80% was used for training.

---

## Modeling Approach

### Steps:
1. **Data Preprocessing:**
   - Loaded the dataset, cleaned missing data, and encoded categorical features.
   - Created edges between flights based on temporal and spatial proximity (origin/destination airport).
  
2. **Graph Construction:**
   - Created a graph where flights were represented as nodes and relationships as edges.
   - Node features and edge indices were fed into the GNN model.

3. **Model Training:**
   - Utilized a 2-layer Graph Convolutional Network (GCN) for node classification.
   - The GNN was trained using node features and edge relationships to predict flight delays.

4. **Model Evaluation:**
   - Evaluated the GNN model on the test set using accuracy, precision, recall, F1-score, and ROC curve.

---

## GNN Model Architecture

- **Input Layer:** Encoded node features representing flights.
- **Graph Convolutional Layers (2 layers):** These layers aggregate information from neighboring nodes (flights) to predict the delay status.
- **Output Layer:** Predicts whether a flight will be delayed (binary classification).

### Hyperparameters:
- **Learning Rate:** 0.01
- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy Loss
- **Epochs:** 200
- **Hidden Dimensions:** 64

---

## Results

### Confusion Matrix
- **True Positives (Delayed Flights Correctly Predicted):** 1068
- **True Negatives (On-time Flights Correctly Predicted):** 29291
- **False Positives (On-time Flights Incorrectly Predicted as Delayed):** 551
- **False Negatives (Delayed Flights Incorrectly Predicted as On-time):** 671

### Performance Metrics:
- **Precision:** 96.09% 
- **Recall:** 96.13% 
- **F1 Score:** 96.07%

---

## Conclusion

This project demonstrates that GNNs are highly effective for flight delay prediction, achieving over 96% in both precision and recall. By leveraging both the flight-specific features and the network structure of flight schedules, the GNN model outperformed traditional machine learning models.

### Benefits of GNN:
- Captures flight dependencies and relationships, offering better accuracy in predicting delays.
- Useful for operational decision-making in airline management.

### Limitations:
- GNNs require more computational power compared to traditional models.
- Building and training GNN models are complex, especially for very large datasets.
