"""
DS-37 — El Yazısı Rakam Tanıma (SVM + GridSearchCV) — CAPSTONE

Bir posta/form okuma sistemi için el yazısı rakamları (0-9) tanıyan bir
model kuracaksın. Veri 8×8 piksellik küçük gri ton görüntülerden oluşuyor
(her görüntü 64 sayıya açılmış). 64 feature → yüksek boyut → SVM'in
parladığı yer. SVM mesafe/iç-çarpım tabanlı olduğu için scaling ZORUNLU.

Bu CAPSTONE'da ML-04'ün HEPSİNİ kullanıyorsun:
- SVM kernel seçimi (linear / rbf / poly)
- C ve gamma tuning → GridSearchCV
- Cross-validation (StratifiedKFold)
- 5 algoritmayı (NB / DT / RF / SVM / LR) yarıştırıp en iyisini seçme

⚠️ ÖNEMLİ: GridSearchCV ve hiçbir yerde n_jobs=-1 KULLANMA. n_jobs=1 yaz
(veya hiç yazma). macOS'ta paralel worker (loky) çökebiliyor.

Her fonksiyonun pass kısmını doldur. Testleri çalıştır, hepsi geçene kadar
iterate et: `python watch.py` veya `pytest tests/test_question.py -v`
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Veri setini yükle
def load_digits_data():
    """
    sklearn'in built-in Digits veri setini yükle.

    Returns:
        tuple: (X, y)
            X: np.ndarray, shape (1797, 64) — her satır 8×8 görüntünün
               64 piksele açılmış hali (gri ton değerleri 0-16)
            y: np.ndarray, shape (1797,) — rakam etiketleri (0-9)

    İpucu:
    - from sklearn.datasets import load_digits
    - return load_digits(return_X_y=True)
    """
    X, y = load_digits(return_X_y=True)
    return X, y


# 2. Veriyi keşfet
def explore_data(X, y):
    """
    Veri seti hakkında özet bilgi çıkar.

    Args:
        X, y: load_digits_data'dan dönen veriler

    Returns:
        dict: {
            'n_samples': int (1797),
            'n_features': int (64),
            'n_classes': int (10),
            'image_shape': (8, 8),
            'class_distribution': dict {0: int, 1: int, ..., 9: int}
        }

    İpucu:
    - X.shape[0] → örnek sayısı, X.shape[1] → feature sayısı
    - import numpy as np
    - classes, counts = np.unique(y, return_counts=True)
    - class_distribution = {int(c): int(n) for c, n in zip(classes, counts)}
    """
    classes, counts = np.unique(y, return_counts=True)
    return {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(classes),
        'image_shape': (8, 8),
        'class_distribution': {int(c): int(n) for c, n in zip(classes, counts)},
    }


# 3. Örnek bir rakamı görselleştir
def visualize_sample(X, y, index=0):
    """
    Tek bir örneği (X[index]) 8×8 görüntüye geri çevirip çiz.

    Args:
        X, y: veri seti
        index: gösterilecek örneğin indeksi (default 0)

    Returns:
        None (sadece matplotlib ile çizim yapar)

    İpucu:
    - import matplotlib.pyplot as plt
    - image = X[index].reshape(8, 8)
    - plt.imshow(image, cmap='gray')
    - plt.title(f"Etiket: {y[index]}")
    - plt.axis('off')
    - (test ortamında pencere açmaması için return None yeterli)
    """
    image = X[index].reshape(8, 8)
    plt.imshow(image, cmap='gray')
    plt.title(f"Etiket: {y[index]}")
    plt.axis('off')
    return None


# 4. Train / test böl
def split_data(X, y):
    """
    Veriyi train/test olarak böl.

    - test_size = 0.2
    - stratify = y (sınıf oranları korunsun)
    - random_state = 42

    Returns:
        tuple: (X_train, X_test, y_train, y_test)

    İpucu:
    - from sklearn.model_selection import train_test_split
    - return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    """
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# 5. SVM Pipeline kur
def build_svm_pipeline(kernel='rbf', C=1, gamma='scale'):
    """
    StandardScaler + SVC'yi tek bir Pipeline içinde birleştir.

    SVM iç-çarpım/mesafe tabanlı olduğu için scaling ZORUNLU. Pipeline
    kullanmamız, CV/fit sırasında scaler'ın yalnızca train kısmıyla fit
    edilmesini (data leakage engelini) otomatik garanti eder.

    Args:
        kernel: 'rbf' / 'linear' / 'poly' (default 'rbf')
        C: ceza parametresi (default 1)
        gamma: kernel katsayısı (default 'scale')

    Returns:
        sklearn.pipeline.Pipeline: adımlar
            [('scaler', StandardScaler()),
             ('svm', SVC(kernel=kernel, C=C, gamma=gamma))]

    İpucu:
    - from sklearn.preprocessing import StandardScaler
    - from sklearn.svm import SVC
    - from sklearn.pipeline import Pipeline
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel, C=C, gamma=gamma)),
    ])


# 6. Modeli eğit
def train_model(pipe, X_train, y_train):
    """
    Pipeline'ı train datasıyla fit et.

    Returns:
        sklearn.pipeline.Pipeline: eğitilmiş pipeline

    İpucu: pipe.fit(X_train, y_train); return pipe
    """
    pipe.fit(X_train, y_train)
    return pipe


# 7. Modeli değerlendir
def evaluate_model(pipe, X_test, y_test):
    """
    Test seti üzerinde tahmin yap ve metrikleri hesapla.

    Bu çok sınıflı (10 sınıf) bir problem → f1'i 'macro' average ile al
    (her sınıfa eşit ağırlık verir).

    Returns:
        dict: {
            'accuracy': float,
            'f1_macro': float,
            'confusion_matrix': np.ndarray (shape (10, 10))
        }

    İpucu:
    - from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    - y_pred = pipe.predict(X_test)
    - f1_score(y_test, y_pred, average='macro')
    """
    y_pred = pipe.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }



# 8. Kernel'leri karşılaştır
def compare_kernels(X_train, X_test, y_train, y_test):
    """
    linear / rbf / poly kernel'leri tek tek eğitip test accuracy'lerini
    karşılaştır.

    Returns:
        dict: {kernel_adi: test_accuracy} — örn.
              {'linear': 0.97, 'rbf': 0.98, 'poly': 0.98}

    İpucu:
    - her kernel için build_svm_pipeline(kernel=k) kur, fit et,
      X_test üzerinde accuracy_score hesapla.
    """
    results = {}
    for kernel in ['linear', 'rbf', 'poly']:
        pipe = build_svm_pipeline(kernel=kernel)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[kernel] = accuracy_score(y_test, y_pred)
    return results


# 9. GridSearchCV ile C × gamma tuning
def grid_search_svm(X_train, y_train):
    """
    SVM pipeline'ı üzerinde GridSearchCV çalıştır.

    - pipe = build_svm_pipeline()  (rbf kernel)
    - param_grid = {
          'svm__C': [1, 10],
          'svm__gamma': ['scale', 0.01, 0.001],
      }
    - cv = 5, scoring = 'accuracy'
    - n_jobs = 1   ⚠️ macOS loky çökmesini önlemek için MUTLAKA 1!

    Returns:
        fitted GridSearchCV nesnesi

    İpucu:
    - from sklearn.model_selection import GridSearchCV
    - Pipeline adımı 'svm' olduğu için parametre öneki 'svm__'
    - grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=1)
    - grid.fit(X_train, y_train); return grid
    """
    pipe = build_svm_pipeline()
    param_grid = {
        'svm__C': [1, 10],
        'svm__gamma': ['scale', 0.01, 0.001],
    }
    grid = GridSearchCV(
        pipe, param_grid, cv=5, scoring='accuracy', n_jobs=1,
    )
    grid.fit(X_train, y_train)
    return grid


# 10. En iyi parametreleri al
def get_best_params(grid):
    """
    Fitted GridSearchCV'den en iyi parametre kombinasyonunu dön.

    Returns:
        dict: grid.best_params_ (örn. {'svm__C': 10, 'svm__gamma': 'scale'})
    """
    return grid.best_params_


# 11. Cross-validation
def cross_validate_model(pipe, X, y, cv=5):
    """
    Pipeline'ı StratifiedKFold cross-validation ile değerlendir (accuracy).

    Returns:
        dict: {
            'mean': float (ortalama CV accuracy),
            'std': float (standart sapma)
        }

    İpucu:
    - from sklearn.model_selection import cross_val_score, StratifiedKFold
    - skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    - scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
    - return {'mean': float(scores.mean()), 'std': float(scores.std())}
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
    return {
        'mean': float(scores.mean()),
        'std': float(scores.std()),
    }


# 12. 5 algoritmayı karşılaştır
def compare_five_algorithms(X_train, X_test, y_train, y_test):
    """
    5 farklı algoritmayı aynı veri üzerinde yarıştır:
        NB  → GaussianNB
        DT  → DecisionTreeClassifier(random_state=42)
        RF  → RandomForestClassifier(random_state=42)   (n_jobs VERME!)
        SVM → SVC(kernel='rbf', C=10, gamma='scale')
        LR  → LogisticRegression(max_iter=5000)

    Her algoritmayı StandardScaler ile bir Pipeline içinde kur (scaling
    tüm mesafe/gradyan tabanlı modellere yardımcı olur), eğit, test
    accuracy'sini ölç.

    ⚠️ Hiçbir modelde n_jobs=-1 kullanma (macOS loky sorunu).

    Returns:
        list of dict: her eleman {'model': str, 'test_accuracy': float}
        (DataFrame de döndürebilirsin ama list of dict en basiti.)

    İpucu:
    - from sklearn.naive_bayes import GaussianNB
    - from sklearn.tree import DecisionTreeClassifier
    - from sklearn.ensemble import RandomForestClassifier
    - from sklearn.linear_model import LogisticRegression
    - models = {'NB': GaussianNB(), 'DT': ..., 'RF': ..., 'SVM': ..., 'LR': ...}
    - her model için Pipeline([('scaler', StandardScaler()), ('model', m)])
    """
    models = {
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(random_state=42),
        'RF': RandomForestClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale'),
        'LR': LogisticRegression(max_iter=5000),
    }
    results = []
    for name, model in models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results.append({
            'model': name,
            'test_accuracy': accuracy_score(y_test, y_pred),
        })
    return results


# 13. Tek görüntü için tahmin
def predict_digit(pipe, image_row):
    """
    Tek bir 64-boyutlu görüntü satırı için rakam tahmini yap.

    Args:
        pipe: eğitilmiş pipeline
        image_row: 64 elemanlı 1D array/list (tek görüntünün piksel değerleri)

    Returns:
        dict: {'predicted_digit': int}

    İpucu:
    - import numpy as np
    - X_new = np.array(image_row).reshape(1, -1)
    - pred = int(pipe.predict(X_new)[0])
    """
    X_new = np.array(image_row).reshape(1, -1)
    pred = int(pipe.predict(X_new)[0])
    return {'predicted_digit': pred}


# 14. GridSearch best model'i test'te değerlendir
def evaluate_at_best(X_train, X_test, y_train, y_test):
    """
    grid_search_svm ile en iyi modeli bul ve test setinde değerlendir.

    Returns:
        dict: {
            'best_params': dict (grid.best_params_),
            'test_accuracy': float (en iyi modelin test accuracy'si)
        }

    İpucu:
    - grid = grid_search_svm(X_train, y_train)
    - best_model = grid.best_estimator_
    - acc = accuracy_score(y_test, best_model.predict(X_test))
    """
    grid = grid_search_svm(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    return {
        'best_params': grid.best_params_,
        'test_accuracy': accuracy_score(y_test, y_pred),
    }


# 15. Tüm pipeline'ı uçtan uca çalıştır
def run_pipeline():
    """
    Yukarıdaki fonksiyonları birleştirip tam akışı çalıştır:
    1. Veri yükle + keşfet
    2. Train/test böl
    3. Kernel'leri karşılaştır → en iyi kernel
    4. GridSearchCV → best_score_ + best model test accuracy
    5. 5 algoritmayı yarıştır → en iyi algoritma

    Returns:
        dict: {
            'n_samples': int (1797),
            'n_classes': int (10),
            'best_kernel': str (compare_kernels'te en yüksek),
            'grid_best_score': float (grid.best_score_),
            'grid_test_accuracy': float (best model'in test accuracy'si),
            'best_algorithm': str (5'li kıyasta en yüksek accuracy'li model)
        }
    """
    X, y = load_digits_data()
    info = explore_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Kernel karşılaştırması
    kernel_scores = compare_kernels(X_train, X_test, y_train, y_test)
    best_kernel = max(kernel_scores, key=kernel_scores.get)

    # GridSearch
    grid = grid_search_svm(X_train, y_train)
    grid_best_score = float(grid.best_score_)
    best_model = grid.best_estimator_
    grid_test_accuracy = accuracy_score(y_test, best_model.predict(X_test))

    # 5-algoritma karşılaştırması
    five = compare_five_algorithms(X_train, X_test, y_train, y_test)
    best_algorithm = max(five, key=lambda d: d['test_accuracy'])['model']

    return {
        'n_samples': info['n_samples'],
        'n_classes': info['n_classes'],
        'best_kernel': best_kernel,
        'grid_best_score': grid_best_score,
        'grid_test_accuracy': grid_test_accuracy,
        'best_algorithm': best_algorithm,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 Pipeline Sonuçları:")
    print(f"  Örnek sayısı       : {result['n_samples']}")
    print(f"  Sınıf sayısı       : {result['n_classes']}")
    print(f"  En iyi kernel      : {result['best_kernel']}")
    print(f"  Grid best CV score : {result['grid_best_score']:.4f}")
    print(f"  Grid test accuracy : {result['grid_test_accuracy']:.4f}")
    print(f"  En iyi algoritma   : {result['best_algorithm']}")
