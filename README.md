===TP4 – RAG avec LangChain4j & Google Gemini===

**Contexte du projet**

Ce projet Java illustre les différentes phases du RAG (Retrieval Augmented Generation) à l’aide de la bibliothèque LangChain4j et du modèle Gemini 2.5-Flash de Google.

Il a été réalisé dans le cadre du cours “Fine-tuning et RAG” (UCA / EMSI – Prof. Richard Grin).

Le RAG permet à un modèle de langage (LM) d’accéder à des sources externes fiables (PDF, Web, etc.) pour enrichir ses réponses sans modifier ses paramètres internes.

Le but est de réduire les hallucinations, améliorer la précision, et maintenir les connaissances à jour.

**Technologies utilisées**

Composant	Description

LangChain4j 1.8.0	Framework Java pour orchestrer les pipelines RAG

LangChain4j Gemini (1.2.0)	Connecteur pour Google Gemini

AllMiniLmL6V2 ONNX	Modèle d’embedding léger pour la recherche sémantique

Apache Tika Parser	Extraction de texte depuis les documents PDF

Tavily Web Search Engine	Moteur de recherche Web intégré au RAG

SLF4J	Gestion des logs

Java 21 + Maven	Environnement d’exécution

**Architecture du projet**

Le projet est organisé en plusieurs tests progressifs :

1) test1 – RAG Naïf

Implémente les deux phases principales du RAG :

Phase 1 : découpage du PDF rag.pdf, génération des embeddings, stockage en mémoire.

Phase 2 : interrogation du modèle Gemini avec un EmbeddingStoreContentRetriever.

2) test2 – RAG avec logs et améliorations

Activation des logs détaillés LangChain4j (niveau FINE).

Vérification de la clé d’environnement GEMINI_KEY.

Configuration du modèle : gemini-2.5-flash, température 0.3.

3) test3 – Routage de requêtes

Le projet ajoute un mécanisme de routage intelligent :

Plusieurs magasins d’embeddings (rag.pdf, emsi.pdf) sont créés.

Le LanguageModelQueryRouter décide automatiquement vers quelle source diriger la question.

Cela permet de combiner différents domaines : IA/RAG et management/EMSI.


4)  test4 – Pas de RAG (filtrage contextuel)

Introduction d’une classe AssistantSys avec un message système (@SystemMessage) :

! si la question n’est pas liée à l’IA, le modèle répond sans utiliser le RAG.

Intègre un QueryRouter personnalisé pour ignorer les requêtes hors domaine.

5)  test5 – RAG avec récupération Web (Tavily)

Dernière étape : combinaison du RAG local et de la recherche Web.

Création d’un WebSearchEngine Tavily (clé API TAVILY_API_KEY).

Utilisation d’un DefaultQueryRouter combinant le PDF local et Internet.

Le modèle Gemini puise à la fois dans les connaissances du PDF et les résultats Web récents.


** Variables d’environnement **

GEMINI_KEY	Clé API Google Gemini

TAVILY_API_KEY	Clé API Tavily Web Search 
