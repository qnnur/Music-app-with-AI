import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from mlxtend.frequent_patterns import apriori, association_rules

class MusicRecommender:
    def __init__(self):
        self.songs = [
            "Shape of You - Ed Sheeran",
            "Blinding Lights - The Weeknd",
            "Levitating - Dua Lipa",
            "Stay - Justin Bieber & The Kid LAROI",
            "Good 4 U - Olivia Rodrigo",
            "Save Your Tears - The Weeknd",
            "Industry Baby - Lil Nas X & Jack Harlow",
            "Peaches - Justin Bieber",
            "Watermelon Sugar - Harry Styles",
            "Bad Habit - Steve Lacy"
        ]
        
        # Генерация синтетических данных
        np.random.seed(42)
        self.X = np.random.randint(1, 11, size=(500, 10))  # 500 пользователей
        self.y_like = (self.X > 7).astype(int)  # Порог "нравится" - 8+
        
        # Обучение всех моделей
        self.models = {
            # Регрессия (предсказание оценок)
            "Linear Regression": LinearRegression().fit(self.X, self.X),
            "KNN Regressor": KNeighborsRegressor(n_neighbors=5).fit(self.X, self.X),
            
            # Классификация (нравится/не нравится)
            "Logistic Regression": MultiOutputClassifier(LogisticRegression(max_iter=1000)).fit(self.X, self.y_like),
            "Decision Tree": MultiOutputClassifier(DecisionTreeClassifier()).fit(self.X, self.y_like),
            "Random Forest": MultiOutputClassifier(RandomForestClassifier()).fit(self.X, self.y_like),
            "Naive Bayes": MultiOutputClassifier(GaussianNB()).fit(self.X, self.y_like),
            "SVM": MultiOutputClassifier(SVC()).fit(self.X, self.y_like),
            "Gradient Boosting": MultiOutputClassifier(GradientBoostingClassifier()).fit(self.X, self.y_like),
            
            # Дополнительные алгоритмы
            "K-means": KMeans(n_clusters=3).fit(self.X),
            "PCA": PCA(n_components=2).fit(self.X)
        }
        
        # Подготовка данных для Apriori
        self.frequent_itemsets = self._prepare_apriori()
        
        # Метрики моделей
        self.metrics = self._calculate_metrics()
    
    def _prepare_apriori(self):
        """Подготовка данных для алгоритма Apriori"""
        binary_data = (self.X > 7).astype(int)
        df = pd.DataFrame(binary_data, columns=[f"song_{i}" for i in range(10)])
        return apriori(df, min_support=0.1, use_colnames=True)
    
    def _calculate_metrics(self):
        metrics = {}
        for name, model in self.models.items():
            if "Regression" in name or "Regressor" in name:
                pred = model.predict(self.X)
                metrics[name] = {
                    "MSE": mean_squared_error(self.X, pred),
                }
            elif name not in ["K-means", "PCA"]:
                pred = model.predict(self.X)
                metrics[name] = {
                    "Accuracy": accuracy_score(self.y_like, pred)
                }
        return metrics
    
    def predict_ratings(self, user_ratings):
        """Предсказание недостающих оценок"""
        input_data = np.array([r if r is not None else 0 for r in user_ratings]).reshape(1, -1)
        results = {}
        
        for name in ["Linear Regression", "KNN Regressor"]:
            pred = self.models[name].predict(input_data)[0]
            for i, rating in enumerate(user_ratings):
                if rating is None:
                    results[f"{name} - {self.songs[i]}"] = f"{pred[i]:.1f}"
        
        return results
    
    def predict_preferences(self, user_ratings):
        """Предсказание 'нравится/не нравится'"""
        input_data = np.array([r if r is not None else 0 for r in user_ratings]).reshape(1, -1)
        results = {}
        
        for name, model in self.models.items():
            if name not in ["Linear Regression", "KNN Regressor", "K-means", "PCA"]:
                pred = model.predict(input_data)[0]
                for i, rating in enumerate(user_ratings):
                    if rating is None:
                        status = "Нравится" if pred[i] == 1 else "Не нравится"
                        results[f"{name} - {self.songs[i]}"] = status
        
        return results
    
    def analyze_clusters(self, user_ratings):
        """Анализ кластеров"""
        input_data = np.array([r if r is not None else 0 for r in user_ratings]).reshape(1, -1)
        cluster = self.models["K-means"].predict(input_data)[0]
        return f"Ваш музыкальный вкус относится к кластеру {cluster + 1}"
    
    def analyze_pca(self, user_ratings):
        """Анализ главных компонент"""
        input_data = np.array([r if r is not None else 0 for r in user_ratings]).reshape(1, -1)
        components = self.models["PCA"].transform(input_data)[0]
        return f"Главные компоненты: {components[0]:.2f}, {components[1]:.2f}"
    
    def find_association_rules(self):
        """Поиск ассоциативных правил"""
        rules = association_rules(self.frequent_itemsets, metric="lift", min_threshold=1.2)
        return rules.to_dict('records')