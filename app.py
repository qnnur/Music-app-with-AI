from flask import Flask, render_template, request
from models import MusicRecommender

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Инициализация системы рекомендаций
recommender = MusicRecommender()

@app.route("/", methods=["GET", "POST"])
def index():
    results = {
        "ratings": None,
        "preferences": None,
        "cluster": None,
        "pca": None,
        "associations": None
    }
    error = None
    
    if request.method == "POST":
        try:
            # Получаем оценки пользователя
            user_ratings = [request.form.get(f"song_{i}", type=int) for i in range(10)]
            
            if "predict_ratings" in request.form:
                results["ratings"] = recommender.predict_ratings(user_ratings)
            
            elif "predict_preferences" in request.form:
                results["preferences"] = recommender.predict_preferences(user_ratings)
            
            elif "analyze_cluster" in request.form:
                results["cluster"] = recommender.analyze_clusters(user_ratings)
            
            elif "analyze_pca" in request.form:
                results["pca"] = recommender.analyze_pca(user_ratings)
                
            elif "find_associations" in request.form:
                results["associations"] = recommender.find_association_rules()
                
        except Exception as e:
            error = f"Ошибка: {str(e)}"
    
    return render_template(
        "index.html",
        songs=recommender.songs,
        metrics=recommender.metrics,
        results=results,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)