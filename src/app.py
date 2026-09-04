import pandas as pd
from transformers import pipeline
import os

def map_sentiment(stars):
    if stars <= 2:
        return 'Negativo'
    elif stars == 3:
        return 'Neutral'
    else:
        return 'Positivo'

def main():
    print("Iniciando tubería de inferencia de sentimiento...")
    
    # 1. Cargar datos
    input_path = os.path.join('..', 'data', 'raw', 'reviews.csv')
    output_path = os.path.join('..', 'data', 'processed', 'reviews_with_sentiment.csv')
    
    df = pd.read_csv(input_path)
    print(f"Cargadas {len(df)} reseñas.")
    
    # 2. Cargar pipeline de inferencia EXACTAMENTE UNA VEZ
    print("Cargando modelo nlptown/bert-base-multilingual-uncased-sentiment...")
    sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    
    # 3. Ejecutar inferencia
    print("Ejecutando inferencia (esto puede tomar unos momentos)...")
    texts = df['review_text'].astype(str).tolist()
    
    # Manejando truncamiento a 512 tokens
    results = sentiment_pipeline(texts, truncation=True, max_length=512)
    
    # 4. Parsear resultados
    predicted_stars = []
    confidence_scores = []
    sentiment_bands = []
    
    for res in results:
        # res['label'] is typically '1 star', '2 stars', etc.
        star_num = int(res['label'].split()[0])
        predicted_stars.append(star_num)
        confidence_scores.append(res['score'])
        sentiment_bands.append(map_sentiment(star_num))
        
    df['predicted_stars'] = predicted_stars
    df['confidence_score'] = confidence_scores
    df['sentiment_band'] = sentiment_bands
    
    # 5. Guardar archivo procesado
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Procesamiento completado. Archivo guardado en: {output_path}")

if __name__ == "__main__":
    # Ensure working directory is correct (script location)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
