import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class ANN:
    def __init__(self, input_size, learning_rate=0.1):
        self.layers = [input_size]
        self.weights = []
        self.biases = []
        self.activations = []
        self.learning_rate = learning_rate
        self.loss_history = []

    def add_layer(self, size, activation):
        prev_size = self.layers[-1]
        self.layers.append(size)
        self.weights.append(np.random.randn(size, prev_size) * np.sqrt(2.0 / (size + prev_size)))
        self.biases.append(np.zeros((size, 1)))
        self.activations.append(activation)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return x * (1 - x)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_deriv(self, x):
        return (x > 0).astype(float)

    def _activate(self, x, func):
        return self._sigmoid(x) if func == 'sigmoid' else self._relu(x)

    def _activate_deriv(self, x, func):
        return self._sigmoid_deriv(x) if func == 'sigmoid' else self._relu_deriv(x)

    def feed_forward(self, x):
        outputs = [np.array(x).reshape(-1, 1)]
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], outputs[-1]) + self.biases[i]
            a = self._activate(z, self.activations[i])
            outputs.append(a)
        return outputs

    def train(self, x, y):
        outputs = self.feed_forward(x)
        targets = np.array(y).reshape(-1, 1)
        errors = [targets - outputs[-1]]

        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(self.weights[i].T, errors[0])
            errors.insert(0, delta)

        for i in range(len(self.weights)):
            grad = errors[i] * self._activate_deriv(outputs[i+1], self.activations[i])
            self.weights[i] += self.learning_rate * np.dot(grad, outputs[i].T)
            self.biases[i] += self.learning_rate * grad

        loss = np.mean((targets - outputs[-1]) ** 2)
        self.loss_history.append(loss)

    def predict(self, x):
        output = self.feed_forward(x)[-1]
        return 1 if output[0][0] >= 0.5 else 0, float(output[0][0])

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

def evaluasi_model(model, X_tes, y_tes):
    prediksi = []
    probabilitas = []

    for x in X_tes:
        pred, prob = model.predict(x)
        prediksi.append(pred)
        probabilitas.append(prob)

    akurasi = accuracy_score(y_tes, prediksi)
    presisi = precision_score(y_tes, prediksi)
    recall = recall_score(y_tes, prediksi)
    f1 = f1_score(y_tes, prediksi)
    cm = confusion_matrix(y_tes, prediksi)
    TP, FN, FP, TN = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrik = {
        'akurasi': akurasi,
        'presisi': presisi,
        'recall': recall,
        'f1_score': f1,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }

    return metrik, probabilitas

def main():
    df = pd.read_csv('cleaned_data.csv')

    fitur_cols = ["age","gender","jundice","autism","A1_Score","A2_Score","A3_Score",
                  "A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",]
    fitur = df[fitur_cols]
    target = df['class/ASD']

    fitur['age'] = fitur['age'].astype(int)

    scaler = StandardScaler()
    fitur_scaled = scaler.fit_transform(fitur)

    X = fitur_scaled
    y = target.values
    print(f"Jumlah data awal: {len(y)}")
    X_latih, X_tes, y_latih, y_tes = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(k_neighbors=3, random_state=42)
    X_latih, y_latih = smote.fit_resample(X_latih, y_latih)
    print(f"Jumlah data setelah SMOTE pada data latih: {len(y_latih)}")

    model = ANN(input_size=X_latih.shape[1], learning_rate=0.01)
    model.add_layer(8, activation='relu')
    model.add_layer(4, activation='relu')
    model.add_layer(1, activation='sigmoid')

    for epoch in range(10):
        for x, y_benar in zip(X_latih, y_latih):
            model.train(x, y_benar)
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/10, Loss: {model.loss_history[-1]:.6f}")

    # model.save_model('ANN_model.pkl')
    # print("\nModel telah disimpan ke 'ANN_model.pkl'")

    metrik, probabilitas = evaluasi_model(model, X_tes, y_tes)
    print("\nHasil Evaluasi:")
    for k, v in metrik.items():
        print(f"{k.capitalize()}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main() 
