import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from google import genai
from typing import List

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Pon tu GEMINI_API_KEY en .env")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

CSV_PATH = os.path.join("data", "data.csv")
TOP_K = 5  # Aumentado para tener más contexto


# ============================
# CARGA DEL CSV (MEJORADO)
# ============================
def load_csv_and_build_index(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No existe {csv_path}")

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    
    # Crear textos enriquecidos con descripciones legibles
    enriched_texts = []
    for _, row in df.iterrows():
        parts = []
        for col, val in row.items():
            # Hacer los nombres de columnas más legibles
            readable_col = col.replace("_", " ").replace("customerID", "cliente ID")
            parts.append(f"{readable_col}: {val}")
        
        # Agregar descripciones adicionales para mejorar búsqueda
        text = " | ".join(parts)
        
        # Agregar términos adicionales para facilitar búsqueda
        if row.get('Churn') == '1':
            text += " | cliente canceló servicio abandonó"
        if row.get('SeniorCitizen') == '1':
            text += " | adulto mayor senior"
        if row.get('gender') == '1':
            text += " | mujer femenino"
        elif row.get('gender') == '0':
            text += " | hombre masculino"
        if 'Fiber optic' in text:
            text += " | fibra óptica internet rápido"
        if 'InternetService' in text:
            text += " | servicio internet conexión"
        
        enriched_texts.append(text)
    
    # TF-IDF con configuración más permisiva
    vectorizer = TfidfVectorizer(
        stop_words=None,  # No eliminar palabras
        min_df=1,  # Considerar todas las palabras
        ngram_range=(1, 2),  # Incluir bigramas
        lowercase=True
    )
    vectors = vectorizer.fit_transform(enriched_texts)

    return df, enriched_texts, vectorizer, vectors


print("Cargando CSV e indexando...")
DF, TEXTS, VECTORIZER, VECTORS = load_csv_and_build_index(CSV_PATH)
print(f"Indexado {len(TEXTS)} registros del CSV.")


def retrieve_top_k(query: str, k: int = TOP_K) -> List[str]:
    # Expandir la query con sinónimos comunes
    query_expanded = query.lower()
    
    # Agregar términos relacionados
    if 'cancelar' in query_expanded or 'cancelación' in query_expanded:
        query_expanded += " churn abandonar servicio"
    if 'mayor' in query_expanded or 'adulto' in query_expanded:
        query_expanded += " senior citizen"
    if 'fibra' in query_expanded or 'internet' in query_expanded:
        query_expanded += " fiber optic service"
    if 'mujer' in query_expanded or 'hombre' in query_expanded:
        query_expanded += " gender"
    if 'precio' in query_expanded or 'costo' in query_expanded or 'pago' in query_expanded:
        query_expanded += " charges payment monthly"
    if 'cliente' in query_expanded or 'cuantos' in query_expanded:
        query_expanded += " customer customerID total count"
    
    q_vec = VECTORIZER.transform([query_expanded])
    cosine_similarities = linear_kernel(q_vec, VECTORS).flatten()
    top_indices = cosine_similarities.argsort()[::-1][:k]

    results = []
    for idx in top_indices:
        score = cosine_similarities[idx]
        results.append(f"Fila {idx+1} (relevancia={score:.3f}):\n{TEXTS[idx]}\n")
    
    # Si no hay resultados o todos tienen score muy bajo, devolver muestra aleatoria
    if not results or max(cosine_similarities) < 0.001:
        import random
        sample_indices = random.sample(range(len(TEXTS)), min(k, len(TEXTS)))
        results = [f"Fila {idx+1} (muestra aleatoria):\n{TEXTS[idx]}\n" for idx in sample_indices]
    
    return results


# ============================
# WEBSOCKET
# ============================
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_msg = await websocket.receive_text()

            # Recuperar contexto
            top_rows = retrieve_top_k(user_msg)
            if not top_rows:
                await websocket.send_text(
                    "No encontré información suficientemente relevante en el CSV para responder tu pregunta. "
                    "Intenta preguntar sobre: clientes, servicios, contratos, cancelaciones, pagos, etc."
                )
                continue

            context_text = "\n".join(top_rows)
            
            system_prompt = (
                "Eres un asistente que analiza datos de clientes de telecomunicaciones.\n"
                "IMPORTANTE: Debes responder SOLO basándote en la información del CSV que te proporciono.\n"
                "Los datos incluyen información sobre clientes, sus servicios, contratos y si cancelaron (Churn).\n\n"
                "Notas sobre los datos:\n"
                "- Churn: 1 = cliente canceló, 0 = cliente activo\n"
                "- SeniorCitizen: 1 = adulto mayor, 0 = no\n"
                "- gender: 1 = mujer, 0 = hombre\n"
                "- MonthlyCharges: cargo mensual en dólares\n"
                "- tenure: meses que el cliente ha estado con la empresa\n\n"
                "Filas relevantes del CSV:\n\n"
            )

            final_prompt = (
                system_prompt
                + context_text
                + f"\n\nPregunta del usuario: {user_msg}\n\n"
                + "Respuesta (analiza los datos y responde de forma clara y concisa):"
            )

            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-2.5-flash",
                    contents=final_prompt
                )

                text = getattr(response, "text", None) \
                    or getattr(response, "output_text", None) \
                    or str(response)

            except Exception as e:
                text = f"Error al llamar a Gemini: {str(e)}"

            await websocket.send_text(text)

    except WebSocketDisconnect:
        print("Cliente desconectado")
    except Exception as e:
        print(f"Error en WebSocket: {str(e)}")
        try:
            await websocket.send_text(f"Error interno: {str(e)}")
        except:
            pass
        await websocket.close()


# ============================
# ENDPOINT OPCIONAL PARA RECARGAR CSV
# ============================
@app.post("/reload-csv")
async def reload_csv():
    global DF, TEXTS, VECTORIZER, VECTORS
    DF, TEXTS, VECTORIZER, VECTORS = load_csv_and_build_index(CSV_PATH)
    return {"status": "ok", "rows_indexed": len(TEXTS)}


# ============================
# RUTA PARA SERVIR index.html
# ============================
@app.get("/")
async def read_index():
    return FileResponse("frontend/index.html")


# ============================
# STATIC FILES (DEBE IR AL FINAL)
# ============================
app.mount("/", StaticFiles(directory="frontend"), name="frontend")