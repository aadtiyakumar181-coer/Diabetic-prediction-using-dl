// Diabetes Prediction using Logistic Regression (from scratch, C++17)
// Dataset: Pima Indians Diabetes (data/diabetes.csv)
// Build:  g++ -std=c++17 -O2 main.cpp -o diabetes
// Run:    ./diabetes data/diabetes.csv

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

struct Dataset {
    vector<vector<double>> X;  // features
    vector<int> y;             // labels (0 = no diabetes, 1 = diabetes)
};

// ---------- Step 1: Load CSV ----------
bool loadCSV(const string& path, Dataset& d) {
    ifstream file(path);
    if (!file) return false;
    string line;
    getline(file, line);  // skip header
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string cell;
        vector<double> row;
        while (getline(ss, cell, ',')) row.push_back(stod(cell));
        if (row.size() != 9) continue;
        d.y.push_back(static_cast<int>(row.back()));
        row.pop_back();
        d.X.push_back(row);
    }
    return !d.X.empty();
}

// ---------- Step 2: Clean data ----------
// In this dataset, 0 in these columns really means "missing".
// Columns: 1 Glucose, 2 BloodPressure, 3 SkinThickness, 4 Insulin, 5 BMI
void fixMissing(Dataset& d) {
    for (int c : {1, 2, 3, 4, 5}) {
        vector<double> valid;
        for (auto& r : d.X) if (r[c] != 0) valid.push_back(r[c]);
        sort(valid.begin(), valid.end());
        double median = valid[valid.size() / 2];
        for (auto& r : d.X) if (r[c] == 0) r[c] = median;
    }
}

// ---------- Step 3: Scale features (standardization) ----------
struct Scaler {
    vector<double> mean, stdev;
    void fit(const vector<vector<double>>& X) {
        size_t f = X[0].size();
        mean.assign(f, 0); stdev.assign(f, 0);
        for (auto& r : X) for (size_t j = 0; j < f; j++) mean[j] += r[j];
        for (auto& m : mean) m /= X.size();
        for (auto& r : X) for (size_t j = 0; j < f; j++) stdev[j] += pow(r[j] - mean[j], 2);
        for (auto& s : stdev) { s = sqrt(s / X.size()); if (s == 0) s = 1; }
    }
    vector<double> transform(const vector<double>& r) const {
        vector<double> o(r.size());
        for (size_t j = 0; j < r.size(); j++) o[j] = (r[j] - mean[j]) / stdev[j];
        return o;
    }
};

// ---------- Step 4: Logistic Regression ----------
double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

struct LogisticRegression {
    vector<double> w;
    double b = 0;

    double predictProb(const vector<double>& x) const {
        double z = b;
        for (size_t j = 0; j < w.size(); j++) z += w[j] * x[j];
        return sigmoid(z);
    }

    void train(const vector<vector<double>>& X, const vector<int>& y,
               double lr = 0.1, int epochs = 1000) {
        size_t n = X.size(), f = X[0].size();
        w.assign(f, 0.0);
        for (int e = 0; e < epochs; e++) {
            vector<double> gw(f, 0.0);
            double gb = 0, loss = 0;
            for (size_t i = 0; i < n; i++) {
                double p = predictProb(X[i]);
                double err = p - y[i];
                for (size_t j = 0; j < f; j++) gw[j] += err * X[i][j];
                gb += err;
                loss += -(y[i] * log(p + 1e-12) + (1 - y[i]) * log(1 - p + 1e-12));
            }
            for (size_t j = 0; j < f; j++) w[j] -= lr * gw[j] / n;
            b -= lr * gb / n;
            if (e % 200 == 0) cout << "Epoch " << e << "  loss = " << loss / n << "\n";
        }
    }
};

// ---------- Step 5: Evaluate ----------
void evaluate(const LogisticRegression& m, const vector<vector<double>>& X, const vector<int>& y) {
    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < X.size(); i++) {
        int pred = m.predictProb(X[i]) >= 0.5 ? 1 : 0;
        if (pred == 1 && y[i] == 1) tp++;
        else if (pred == 0 && y[i] == 0) tn++;
        else if (pred == 1 && y[i] == 0) fp++;
        else fn++;
    }
    double acc = double(tp + tn) / X.size();
    double prec = tp + fp ? double(tp) / (tp + fp) : 0;
    double rec = tp + fn ? double(tp) / (tp + fn) : 0;
    double f1 = prec + rec ? 2 * prec * rec / (prec + rec) : 0;
    cout << "\n--- Test Results ---\n";
    cout << "Confusion matrix:  TP=" << tp << " FP=" << fp << " FN=" << fn << " TN=" << tn << "\n";
    cout << "Accuracy : " << acc * 100 << "%\n";
    cout << "Precision: " << prec << "\nRecall   : " << rec << "\nF1-score : " << f1 << "\n";
}

int main(int argc, char** argv) {
    string path = argc > 1 ? argv[1] : "data/diabetes.csv";
    Dataset data;
    if (!loadCSV(path, data)) {
        cerr << "Could not read " << path << "\n";
        return 1;
    }
    cout << "Loaded " << data.X.size() << " patients\n";
    fixMissing(data);

    // Shuffle and split 80% train / 20% test
    vector<size_t> idx(data.X.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
    mt19937 rng(42);
    shuffle(idx.begin(), idx.end(), rng);
    size_t nTrain = idx.size() * 8 / 10;

    vector<vector<double>> Xtr, Xte;
    vector<int> ytr, yte;
    for (size_t i = 0; i < idx.size(); i++) {
        auto& x = data.X[idx[i]];
        if (i < nTrain) { Xtr.push_back(x); ytr.push_back(data.y[idx[i]]); }
        else            { Xte.push_back(x); yte.push_back(data.y[idx[i]]); }
    }

    // Scale using training data only (avoids data leakage)
    Scaler sc;
    sc.fit(Xtr);
    for (auto& r : Xtr) r = sc.transform(r);
    for (auto& r : Xte) r = sc.transform(r);

    LogisticRegression model;
    model.train(Xtr, ytr);
    evaluate(model, Xte, yte);

    // Try your own patient
    cout << "\nEnter patient data? (y/n): ";
    char ch; cin >> ch;
    if (ch == 'y' || ch == 'Y') {
        const char* names[8] = {"Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                "Insulin", "BMI", "DiabetesPedigree", "Age"};
        vector<double> p(8);
        for (int j = 0; j < 8; j++) { cout << names[j] << ": "; cin >> p[j]; }
        double prob = model.predictProb(sc.transform(p));
        cout << "Diabetes risk: " << prob * 100 << "% -> "
             << (prob >= 0.5 ? "HIGH RISK" : "LOW RISK") << "\n";
        cout << "(Educational project only, not medical advice.)\n";
    }
    return 0;
}
