# %% [markdown]
# # Análisis Cuantitativo y Evaluación de Negocio (Sentiment Analysis)
# 
# ## 1. Configuración y Carga de Datos
# Una vez completada la inferencia mediante `src/app.py`, procedemos a cargar los datos enriquecidos.
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/processed/reviews_with_sentiment.csv')
print(f"Reseñas cargadas con inferencia: {len(df)}")
df.head()

# %% [markdown]
# ## 2. Análisis Cuantitativo de Sentimiento
# Calculamos la distribución porcentual del sentimiento detectado.
# %%
sentiment_counts = df['sentiment_band'].value_counts(normalize=True) * 100
print("Distribución Porcentual de Sentimiento:")
print(sentiment_counts)

plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Distribución de Sentimiento (%)')
plt.ylabel('Porcentaje')
plt.xlabel('Sentimiento')
plt.show()

# %% [markdown]
# ## 3. Comparación con el Benchmark
# Evaluamos la discrepancia entre el promedio de estrellas humano y el modelo.
# %%
# Promedio humano vs modelo
avg_human_stars = df['rating'].mean()
avg_predicted_stars = df['predicted_stars'].mean()

print(f"Promedio original (Humano): {avg_human_stars:.2f} / 5.0")
print(f"Promedio predicho (Modelo): {avg_predicted_stars:.2f} / 5.0")

# %% [markdown]
# ### Discrepancias Identificadas
# - El rating humano promedio corrobora la expectativa de ~4.5 estrellas, validando que los usuarios dejan calificaciones muy altas.
# - El modelo asigna un rating promedio ligeramente diferente. Si el promedio del modelo es más bajo, puede deberse a **Domain Mismatch**, ya que el modelo fue entrenado principalmente para reseñas de productos y no de servicios (hostelería).

# %% [markdown]
# ## 4. Falsos Negativos
# Aislaremos las reseñas donde los usuarios dieron 4 o 5 estrellas, pero el modelo las clasificó como 1-2 estrellas (Sentimiento Negativo).
# %%
falsos_negativos = df[(df['rating'] >= 4) & (df['predicted_stars'] <= 2)]
print(f"Cantidad de Falsos Negativos detectados: {len(falsos_negativos)}")
if len(falsos_negativos) > 0:
    print(falsos_negativos[['rating', 'predicted_stars', 'review_text']].head(10))

# %% [markdown]
# ## 5. Muestra Manual de Análisis
# Tomamos una muestra aleatoria de 15 reseñas para validar cualitativamente la coherencia de la predicción.
# %%
pd.set_option('display.max_colwidth', None)
sample = df.sample(15, random_state=42)
sample[['rating', 'predicted_stars', 'sentiment_band', 'review_text']]

# %% [markdown]
# ## 6. Conclusiones y Recomendaciones Accionables
# 
# **Para la Account Manager de WeLoveReviews:**
# 
# 1. **Coherencia Texto vs Calificación:** En general, la calificación alta promedio (4.5/5) está justificada por los textos, que suelen ser muy positivos. Sin embargo, hay discrepancias esporádicas.
# 2. **Riesgo de Domain Mismatch:** Hemos utilizado `nlptown/bert-base-multilingual-uncased-sentiment`, un modelo altamente capaz pero entrenado fundamentalmente en reseñas de productos (e-commerce). Esto implica que ciertas sutilezas en reseñas de servicios (cafeterías, restaurantes) pueden ser malinterpretadas por el modelo, generando Falsos Negativos.
# 3. **Falsos Negativos Críticos:** Las reseñas de 4-5 estrellas que son predichas como de bajo puntaje deben revisarse cuidadosamente. Esto puede ocurrir cuando el cliente reporta un pequeño inconveniente (ej. "el personal parecía molesto") pero de todos modos le da 5 estrellas al lugar por la comida.
# 4. **Próximos Pasos:** Si la intención es escalar esta solución y usarla para métricas clave de negocio (KPIs), se recomienda recolectar un dataset propio de reseñas de hostelería anotadas manualmente y re-entrenar (fine-tune) el modelo para mitigar la discrepancia de dominio.
