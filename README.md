# 🎬 Fine-tuning et Déploiement d'un Chatbot de Recommandation de Films : Un Parcours d'Apprentissage

## 📚 Table des matières
1. [Introduction](#-introduction)
2. [Genèse et Évolution du Projet](#-genèse-et-évolution-du-projet)
3. [Choix Techniques et Dataset](#-choix-techniques-et-dataset)
4. [Choix des Technologies]()
5. [Processus Itératif de Fine-tuning](#-processus-itératif-de-fine-tuning)
6. [Développement de l'API](#-développement-de-lapi)
7. [Architecture et Déploiement](#-architecture-et-déploiement)
8. [Déploiement sur HuggingFace Hub](#-déploiement-sur-huggingface-hub)
9. [Résultats et Analyses](#-résultats-et-analyses)
10. [Conclusion et Perspectives](#-conclusion-et-perspectives)

## 🎯 Introduction

Ce projet s'inscrit dans le cadre d'un travail pratique sur le fine-tuning et le déploiement d'un modèle de langage pour la recommandation de films. Notre objectif initial était ambitieux : créer un chatbot capable non seulement de suggérer des films pertinents, mais aussi d'engager une conversation naturelle sur le cinéma, comprenant des requêtes complexes comme "j'ai aimé ce film avec telle actrice, peux-tu me proposer un autre ?" ou "ce film m'a plu pour son ambiance, as-tu des suggestions similaires ?". Comme nous le verrons, la réalité du développement nous a amenés à revoir et adapter ces objectifs initiaux.

📌 **Accès au Projet :**
- Interface web : https://chat.pelliculum.fr
- API : https://chat-api.pelliculum.fr
- Docs API : https://chat-api.pelliculum.fr/docs
- Code Frontend : https://github.com/Pelliculum/Pelliculum-CB-Front
- Modèle Hugging-Face : https://huggingface.co/RealDragonMA/Pelliculum-Chatbot

🚀 **Installation et Lancement Local :**

**Frontend (Angular)**

```bash
# Prérequis
npm install -g @angular/cli

# Installation
git clone https://github.com/Pelliculum/Pelliculum-CB-Front
cd Pelliculum-CB-Front
npm install

# Lancement
npm run dev
```

**Backend (FastAPI)**

```bash
# Installation des dépendances
pip install fastapi uvicorn torch transformers requests

# Lancement
python main.py
# ou
uvicorn main:app --reload
```

L'API sera accessible sur http://localhost:8000 et le frontend sur http://localhost:4200.


## 🌱 Genèse et Évolution du Projet

### 🚀 Premiers Pas et Difficultés Initiales

Notre parcours a débuté avec une ambition peut-être démesurée. Nous voulions créer un chatbot capable de comprendre et d'analyser de multiples aspects des films :
- Genres et sous-genres
- Acteurs et réalisateurs
- Budget et recettes
- Notes et critiques
- Ambiance et style

Cette approche s'est rapidement heurtée à des obstacles majeurs. Le modèle produisait des réponses totalement incohérentes, allant jusqu'à générer des recettes de pâtes au milieu d'une discussion sur le cinéma ! Ces premiers échecs nous ont forcés à remettre en question notre approche.

### 💡 Intervention et Guidance du Professeur

Face à ces difficultés, notre professeur est intervenu, ses conseils étaient :

1. **Simplification de l'Objectif** :
   Commencez petit : Avant de vouloir faire un assistant conversationnel complet, il faut s'assurer que le modèle peut faire une tâche simple correctement.

2. **Focus sur les Données** :
   Il nous a également aidés à comprendre que la qualité du fine-tuning dépendait fortement de la clarté et de la consistance des données d'entraînement.

3. **Partage d'Expertise** :
   Il nous a fourni un exemple de code fonctionnel qui nous a servi de base pour la suite du projet.

## 🔧 Choix Techniques et Dataset

### 🤖 Sélection du Modèle

Le choix de **SmolLM2-135M-Instruct** repose sur plusieurs raisons :
- Taille adaptée aux ressources disponibles (pas de traitement d'images)
- Base déjà instruite pour la compréhension du langage (anglais)
- Communauté active et documentation disponible

### 📊 Analyse Approfondie du Dataset

Le dataset "wykonos/movies" a été sélectionné après une analyse détaillée de ses caractéristiques :

```plaintext
Statistiques clés :
- 722,796 entrées
- 19 colonnes de métadonnées
- Couverture temporelle extensive
```

#### Structure des Données :
```plaintext
id: identifiant unique (int64)
title: titre du film (string)
genres: liste des genres (string)
overview: résumé du film (string)
recommendations: liste d'IDs de films recommandés (string)
[...]
```

L'intégration des IDs TMDb s'est révélée cruciale pour la suite du projet, nous permettant d'enrichir les réponses du chatbot avec des informations détaillées et des visuels.

## 🛠️ Choix des Technologies
[![My Skills](https://skillicons.dev/icons?i=fastapi,python,angular,ts,docker,nginx,cloudflare)](https://skillicons.dev)

### Backend (API)
- **FastAPI** : Choisi pour :
  - Sa performance native grâce à Starlette
  - Sa compatibilité excellente avec les modèles HuggingFace (via librairie transformers)
  - Sa documentation automatique (OpenAPI/Swagger)
  - Sa facilité d'intégration avec Pydantic pour la validation des données
  - Facilité du langage Python

### Frontend
- **Angular** : Sélectionné pour :
  - Sa robustesse et sa maturité
  - Son système de composants réutilisables
  - Son excellent support TypeScript
  - Sa performance avec le change detection
  - Tout le groupe connait et sait utiliser Angular

### Déploiement
- **Docker** : Utilisé pour :
  - La facilité de déploiement

- **Nginx** : Choisi comme reverse proxy pour :
  - Facilité de mise en place
  - Sa gestion efficace du CORS
  - Son rôle de reverseproxy avec Docker

- **Cloudflare** : Utilisé pour :
  - La gestion DNS
  - Le SSL gratuit

## 🔄 Processus Itératif de Fine-tuning

### ❌ Phase 1 : Les Premiers Échecs

Notre première approche était trop ambitieuse. Nous avons tenté d'entraîner le modèle sur l'ensemble des métadonnées :

```python
# Premier essai (trop ambitieux)
input_format = """
Titre: {title}
Genres: {genres}
Réalisateur: {director}
Acteurs: {actors}
Budget: {budget}
Note: {rating}
"""
```

Résultats :
- Loss très élevée
- Réponses incohérentes
- Génération de contenu hors sujet (les fameuses recettes de pâtes !)

### ✨ Phase 2 : Simplification et Premiers Succès

Sur conseil du professeur, nous avons drastiquement simplifié notre approche :

```python
# Approche simplifiée
PROMPT_TEMPLATE = """Suggest movies similar to {title}
movie recommendations:"""
```

Le but était simplement que le modèle renvoit la colonne "recommendations" (liste d'id) du film énoncé par l'utilisateur.

Nouveaux problèmes rencontrés :
1. Le modèle générait des IDs aléatoires au lieu d'utiliser ceux du dataset
2. Quand nous avons remplacé les IDs par les titres correspondants, il répondait systématiquement "None" avec une très bonne training loss (<0.01).
- Analyse du problème :<br>
Le dataset contenait de nombreux films sans recommandations (valeur "None" dans la colonne recommendations)<br>
Durant l'entraînement, le modèle a été exposé à de nombreux exemples où la réponse attendue était "None"<br>
Le modèle a donc appris que répondre "None" était une stratégie optimale pour minimiser la loss<br>
Techniquement, le modèle faisait exactement ce qu'on lui avait appris (d'où la bonne loss) mais ce n'était pas le comportement souhaité

- Solution :<br>
Filtrage du dataset pour ne garder que les films ayant des recommandations<br>
Transformation des IDs en titres de films<br>
Limitation à 8 recommandations maximum par film<br>
Vérification et nettoyage des données avant l'entraînement

### 🎯 Phase 3 : Raffinement et Stabilisation

Après 28 runs documentés sur Weights & Biases, nous avons finalement trouvé une configuration stable :

```python
# Configuration finale
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    num_train_epochs=3,
    learning_rate=0.0002,
    # ...
)
```

! Ne pas prendre en compte les valeurs après l'étape 1000, nous utilisons le modèle de cette étape ! (le train a continué car nous n'avons pas spécifié de max_steps dans les training_args, nous voulions voir la suite)

![WanDB train](/images/wandb_train.png)

### Préparation Minutieuse des Données

Notre processus de prétraitement final incluait :
1. Filtrage des films sans recommandations
2. Conversion des IDs en titres
3. Limitation à 8 recommandations maximum
4. Vérification de la cohérence des données

```python
def replace_ids_with_titles(example):
    titles = []
    for movie_id in example["recommendations"].split('-'):
        if id_to_title.get(int(movie_id)):
            titles.append(id_to_title.get(int(movie_id)))
        if len(titles) == 8:
            break
    example["recommendations"] = ", ".join(titles) if titles else None
    return example
```


## ⚙️ Développement de l'API

### 🔌 Architecture FastAPI et Intégration TMDb

L'API a été conçue pour enrichir les recommandations brutes du modèle. Voici le processus complet :

1. **🛠️ Réception de la requête** :
```python
@app.post("/recommend/")
async def get_recommendations(request: MovieRequest):
    messages = [
        {"role": "system", "content": "You are an expert in movie recommendation"},
        {"role": "user", "content": PROMPT_TEMPLATE.format(title=request.title)}
    ]
```

2. **Génération et Post-traitement** :
```python
def search_tmdb_movies(title: str) -> Dict:
    """Recherche un film sur TMDb par titre."""
    url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "language": "fr-FR",
        "include_adult": False
    }
    # ...
```

### Gestion des Cas Limites

L'API gère plusieurs scénarios :
- Films absents de TMDb
- Titres générés incorrectement par le modèle

## 🏗️ Architecture et Déploiement

### 📡 Infrastructure Technique

Notre infrastructure de déploiement :

```plaintext
Architecture :
└── Serveur (4 cores, 4GB RAM)
    ├── Docker
    │   ├── API FastAPI
    │   └── Front Angular
    ├── Nginx (Reverse Proxy)
    └── Cloudflare DNS
```

![Architecture](/images/architecture.png)

### 💻 Interface Utilisateur

L'interface utilisateur reprend les codes des chatbots modernes :
- Design épuré inspiré de ChatGPT
- Barre de saisie en bas de l'écran
- Affichage des recommandations avec posters et métadonnées

![Accueil chat.pelliculum.fr](/images/chat_home.png)
![Suggest movies similar to Fast X](/images/chat_suggest_fast_x.png)
![Movie details](/images/chat_movie_details.png)

## 📦 Déploiement sur HuggingFace Hub

### Push du Modèle
1. **Préparation :**
   ```python
   from huggingface_hub import notebook_login
   notebook_login()
   ```
2. **Configuration :**
   ```python
   hub_model_id = "RealDragonMA/Pelliculum-Chatbot"
   ```
3. **Entraînement avec Push :**
   ```python
   trainer = SFTTrainer(
      # ... autres configs ...
      args=SFTConfig(
         push_to_hub=True,
         hub_model_id=hub_model_id,
      )
   )
   ```
4. **Sauvegarde et Push :**
   ```python
   trainer.push_to_hub(dataset_name=dataset_name)
   ```

### Fusion et Optimisation
```python
# Fusion de l'adaptateur LoRA
model = model.merge_and_unload()

# Sauvegarde du modèle fusionné
output_merged_dir = os.path.join(OUTPUT_DIR, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)

# Push du modèle fusionné
model.push_to_hub(hub_model_id)
```

### Inférence via Text Generation

Notre modèle peut être utilisé via l'API HuggingFace ou directement avec la bibliothèque transformers :

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("RealDragonMA/Pelliculum-Chatbot")
model = AutoModelForCausalLM.from_pretrained("RealDragonMA/Pelliculum-Chatbot")

# Configuration d'inférence optimisée
model = model.eval()
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(
    inputs,
    max_new_tokens=50,
    temperature=0.2,
    top_p=0.45,
    do_sample=True
)
```

## 📊 Résultats et Analyses

### 🎯 Exemples Concrets

#### ✅ Succès
Requête : "Suggest movies similar to Fast X"
```plaintext
Réponse : [Liste des films avec posters et métadonnées]
```

![Suggest movies similar to Fast X](/images/chat_suggest_fast_x.png)

#### ⚠️ Limitations
1. **Temps de Réponse** : 
   - Moyenne de 8 secondes
   - Principalement dû aux requêtes TMDb

2. **Contraintes Linguistiques** :
   - Nécessité de formuler les requêtes en anglais
   - Format spécifique recommandé : "Suggest movies similar to {title}"

3. **Limites du Dataset** :
   - Risque de recommandations aléatoires pour les films absents du dataset
   - Pas de compréhension profonde des préférences utilisateur

## 🎯 Conclusion et Perspectives

### 📝 Leçons Apprises

1. **L'Importance de la Simplicité** :
   Notre parcours nous a appris qu'il vaut mieux exceller dans une tâche simple que d'échouer dans une tâche complexe.

2. **La Valeur de l'Itération** :
   Les 28 runs sur Weights & Biases témoignent de l'importance d'une approche itérative et persistante.

3. **Le Rôle de l'Expertise** :
   L'intervention de notre professeur a été décisive pour rediriger le projet vers une approche viable.

### 🚀 Pistes d'Amélioration

1. **Optimisation Technique** :
   - Mise en place d'un système de cache pour les requêtes TMDb
   - Optimisation du temps de réponse du modèle

2. **Amélioration du Modèle** :
   - Extension du dataset d'entraînement
   - Fine-tuning sur des conversations plus naturelles
   - Support multilingue

3. **Enrichissement Fonctionnel** :
   - Intégration de la compréhension des préférences utilisateur
   - Interface plus riche en fonctionnalités

Ce projet, malgré ses limitations actuelles, démontre la possibilité de créer un système de recommandation de films fonctionnel en utilisant des modèles de langage fine-tunés. Il illustre également l'importance d'une approche progressive et itérative dans le développement d'applications d'intelligence artificielle.