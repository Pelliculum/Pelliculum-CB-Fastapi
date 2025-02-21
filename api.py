from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
import torch
import requests
import json
from typing import List, Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Modèle et configuration
MODEL_ID = "RealDragonMA/Pelliculum-Chatbot"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chargement du modèle et du tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
model = model.eval()

# Clé API TMDb
TMDB_API_KEY = "efc1fdea36e98dc437d419f495a37666"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Template pour les prompts
PROMPT_TEMPLATE = """Suggest movies similar to {title}
movie recommendations:"""

class MovieRequest(BaseModel):
    title: str

def search_tmdb_movies(title: str) -> Dict:
    """Recherche un film sur TMDb par titre."""
    url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "language": "fr-FR",
        "include_adult": False
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200 and response.json().get("results"):
        movie = response.json()["results"][0]  # Prend le premier résultat
        return {
            **movie,
            "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None
        }
    
    return {"id": None, "title": title, "poster": None}  # Retourne juste le titre si rien n’est trouvé

@app.post("/recommend/", 
          response_model=Dict[str, List[Dict]],
          summary="Obtenir des recommandations de films",
          description="""
          Génère une liste de films recommandés basée sur le titre fourni.
          
          Le processus combine :
          1. Génération de recommandations par IA
          2. Enrichissement des données via TMDb
          
          Les résultats incluent les détails complets des films (affiches, notes, etc.)
          """,
          response_description="Liste des films recommandés avec leurs détails complets",
          tags=["Recommendations"])
async def get_recommendations(request: MovieRequest):
    try:
        # Préparation des messages
        messages = [
            {"role": "system", "content": "You are an expert in movie recommendation"},
            {"role": "user", "content": PROMPT_TEMPLATE.format(title=request.title)}
        ]

        # Tokenization et génération
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
        
        outputs = model.generate(
            inputs,
            max_new_tokens=50,
            temperature=0.2,
            top_p=0.45,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )

        # Décodage de la réponse
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        (response.split("\n")[-1])

        # Extraction des titres recommandés (séparés par des virgules)
        recommended_titles = [title.strip() for title in response.split("\n")[-1].split(",") if title.strip()]
        
        # Recherche des films sur TMDb
        recommendations = [search_tmdb_movies(title) for title in recommended_titles]
        
        # Filtrer les recommandations pour ne garder que celles avec un id valide
        filtered_recommendations = [rec for rec in recommendations if rec["id"] is not None]

        return {"recommendations": filtered_recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
