# Instalar el nuevo SDK (en Colab):
# !pip install -q google-genai pymongo PyPDF2

import os
import pymongo
from google import genai
from google.genai import types
from PyPDF2 import PdfReader

# =======================
# CONFIGURACIÓN
# =======================
#GOOGLE_API_KEY = GOOGLE_API_KEY
#MONGODB_URI = MONGODB_URI

if not GOOGLE_API_KEY or not MONGODB_URI:
    raise ValueError("Faltan GOOGLE_API_KEY o MONGODB_URI en las variables de entorno.")

# Cliente del nuevo SDK
client_genai = genai.Client(api_key=GOOGLE_API_KEY)

# Cliente de MongoDB (lo dejo a nivel de módulo para que `procesar_pdf` lo use)
client_mongo = pymongo.MongoClient(MONGODB_URI)
db = client_mongo.pdf_embeddings_db
collection = db.pdf_vectors

# =======================
# LECTURA DEL PDF
# =======================
def leer_pdf(path_pdf):
    reader = PdfReader(path_pdf)
    texto = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texto += page_text + "\n"
    return texto.strip()

# =======================
# EMBEDDINGS (SDK nuevo)
# =======================
def crear_embedding(texto, task_type="RETRIEVAL_DOCUMENT"):
    response = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=texto,
        config=types.EmbedContentConfig(
            task_type=task_type,
        ),
    )
    return response.embeddings[0].values

# =======================
# ÍNDICE VECTORIAL EN ATLAS
# =======================
def crear_indice_vectorial():
    from pymongo.operations import SearchIndexModel

    # Si el índice ya existe, esto fallará — puedes envolverlo en try/except
    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "similarity": "dotProduct",
                    "numDimensions": 3072,   # debe coincidir con el modelo
                }
            ]
        },
        name="vector_index",
        type="vectorSearch",
    )
    try:
        collection.create_search_index(model=search_index_model)
        print("Índice vectorial creado.")
    except Exception as e:
        print(f"Aviso al crear índice (puede que ya exista): {e}")

# =======================
# PROCESO PRINCIPAL
# =======================
def procesar_pdf(ruta_pdf):
    texto = leer_pdf(ruta_pdf)
    if not texto:
        print("El PDF no contiene texto.")
        return

    # Chunking simple por caracteres
    trozos = [texto[i:i+1000] for i in range(0, len(texto), 1000)]

    documentos = []
    for i, chunk in enumerate(trozos):
        embedding = crear_embedding(chunk, task_type="RETRIEVAL_DOCUMENT")
        documentos.append({
            "id": i,
            "texto": chunk,
            "embedding": embedding,
        })

    collection.insert_many(documentos)
    print(f"Se insertaron {len(documentos)} fragmentos con embeddings.")

# =======================
# USO
# =======================
if __name__ == "__main__":
    crear_indice_vectorial()
    procesar_pdf("aws-cloud-adoption-framework_XL.pdf")
    print("✅ Embeddings generados y almacenados en MongoDB Atlas.")
