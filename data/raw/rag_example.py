"""
rag_example.py
==============
Eine einfache, demonstrative RAG-Pipeline in Python.
Geschrieben als Lernbeispiel - kein Produktionscode!

Abhängigkeiten (wenn man es wirklich ausführen will):
    pip install openai chromadb sentence-transformers

Hinweis: Viele Funktionen hier sind pseudo-implementiert oder stark vereinfacht.
Der Fokus liegt auf dem Verständnis der Grundstruktur, nicht auf echter Performance.
"""

import os
import json
# chromadb wäre die echte Vektordatenbank - hier nur als Platzhalter kommentiert
# import chromadb
# from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------
# Konfiguration - in echten Projekten würde das in eine config.yaml
# --------------------------------------------------------------------
CHUNK_SIZE = 300        # Anzahl Wörter pro Chunk (ungefähr)
CHUNK_OVERLAP = 50      # Wörter Overlap zwischen Chunks
TOP_K = 3               # Wie viele Chunks beim Retrieval zurückgegeben werden
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Sentence-Transformer-Modell (Open Source)

# Simulierter "Vektor-Store" als Dictionary - in echt wäre das ChromaDB, Qdrant etc.
VECTOR_STORE = {}


# --------------------------------------------------------------------
# Schritt 1: Dokumente laden
# --------------------------------------------------------------------
def load_documents(file_paths: list) -> list:
    """
    Lädt Textdokumente aus einer Liste von Dateipfaden.
    Gibt eine Liste von Dictionaries zurück: {"filename": ..., "content": ...}

    In einem echten System würde man hier auch PDF-Parser, HTML-Extraktion etc. einbauen.
    """
    documents = []

    for path in file_paths:
        if not os.path.exists(path):
            print(f"[WARN] Datei nicht gefunden: {path} - überspringe.")
            continue

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # minimale Vorverarbeitung: führende/trailing Whitespace entfernen
        content = content.strip()

        documents.append({
            "filename": os.path.basename(path),
            "filepath": path,
            "content": content,
            "char_count": len(content)
        })

        print(f"[INFO] Geladen: {path} ({len(content)} Zeichen)")

    print(f"[INFO] {len(documents)} Dokument(e) geladen.")
    return documents


# --------------------------------------------------------------------
# Schritt 2: Chunking
# --------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """
    Teilt einen Text in überlappende Chunks auf.
    Splitting erfolgt hier wortbasiert (nicht tokenbasiert - das wäre genauer,
    aber für diese Demo reicht Wörter-Splitting).

    Args:
        text: Der zu chunkende Volltext
        chunk_size: Maximale Wortanzahl pro Chunk
        overlap: Überlappung in Wörtern zwischen aufeinanderfolgenden Chunks

    Returns:
        Liste von Chunk-Strings
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        # Nächster Chunk beginnt mit Overlap
        start += chunk_size - overlap

        # Endlosschleifen verhindern (falls overlap >= chunk_size)
        if chunk_size <= overlap:
            print("[ERROR] Overlap darf nicht größer als Chunk-Size sein!")
            break

    return chunks


def chunk_documents(documents: list) -> list:
    """
    Wendet chunk_text() auf alle geladenen Dokumente an.
    Gibt eine flache Liste aller Chunks mit Metadaten zurück.
    """
    all_chunks = []
    chunk_id = 0

    for doc in documents:
        chunks = chunk_text(doc["content"])

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"chunk_{chunk_id:04d}",
                "source_file": doc["filename"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk
            })
            chunk_id += 1

    print(f"[INFO] Insgesamt {len(all_chunks)} Chunks erstellt.")
    return all_chunks


# --------------------------------------------------------------------
# Schritt 3: Embedding erzeugen (simuliert)
# --------------------------------------------------------------------
def embed_text(text: str) -> list:
    """
    Erzeugt einen Embedding-Vektor für den gegebenen Text.

    In echt: SentenceTransformer(EMBEDDING_MODEL).encode(text)
    Hier: Gibt einen Dummy-Vektor zurück (Zufallszahlen) - NUR für Demo!

    Der Vektor hat normalerweise 384 Dimensionen bei MiniLM.
    """
    import random
    # ACHTUNG: Das ist kein echtes Embedding! Nur Placeholder.
    dummy_vector = [random.uniform(-1, 1) for _ in range(384)]
    return dummy_vector


def build_index(chunks: list) -> None:
    """
    Berechnet Embeddings für alle Chunks und speichert sie im globalen VECTOR_STORE.
    In einer echten Implementierung würde man hier ChromaDB.add() oder ähnliches nutzen.
    """
    global VECTOR_STORE

    print("[INFO] Starte Indexierung der Chunks...")

    for chunk in chunks:
        embedding = embed_text(chunk["text"])
        VECTOR_STORE[chunk["chunk_id"]] = {
            **chunk,
            "embedding": embedding
        }

    print(f"[INFO] {len(VECTOR_STORE)} Chunks indiziert.")


# --------------------------------------------------------------------
# Schritt 4: Retrieval
# --------------------------------------------------------------------
def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Berechnet die Kosinus-Ähnlichkeit zwischen zwei Vektoren.
    Manuell implementiert - in echt würde man numpy nutzen.

    cos(θ) = (A · B) / (|A| * |B|)
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a ** 2 for a in vec_a) ** 0.5
    norm_b = sum(b ** 2 for b in vec_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def retrieve(query: str, top_k: int = TOP_K) -> list:
    """
    Findet die relevantesten Chunks für eine gegebene Query.

    Ablauf:
        1. Query embedden
        2. Ähnlichkeit zu allen gespeicherten Vektoren berechnen
        3. Top-k Chunks nach Ähnlichkeit zurückgeben

    In echten Systemen übernimmt das die Vektordatenbank (ANN-Suche, viel schneller).
    Unsere Brute-Force-Suche hier funktioniert, skaliert aber schlecht bei vielen Chunks.
    """
    if not VECTOR_STORE:
        print("[WARN] Kein Index vorhanden. Bitte erst build_index() aufrufen.")
        return []

    query_embedding = embed_text(query)

    # Ähnlichkeit zu jedem Chunk berechnen
    scored_chunks = []
    for chunk_id, chunk_data in VECTOR_STORE.items():
        score = cosine_similarity(query_embedding, chunk_data["embedding"])
        scored_chunks.append((score, chunk_data))

    # Sortieren: höchste Ähnlichkeit zuerst
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Top-k Ergebnisse zurückgeben
    results = []
    for score, chunk in scored_chunks[:top_k]:
        results.append({
            "chunk_id": chunk["chunk_id"],
            "score": round(score, 4),
            "source": chunk["source_file"],
            "text": chunk["text"]
        })

    return results


# --------------------------------------------------------------------
# Schritt 5: Generation (simuliert)
# --------------------------------------------------------------------
def generate_answer(query: str, context_chunks: list) -> str:
    """
    Kombiniert Query und abgerufene Chunks zu einem Prompt und
    schickt ihn an ein LLM.

    Hier nur simuliert - in echt würde man die OpenAI-API oder einen
    lokalen Ollama-Server nutzen.
    """

    # Kontext aus den Chunks zusammensetzen
    context_texts = [f"[Quelle: {c['source']}]\n{c['text']}" for c in context_chunks]
    context_string = "\n\n---\n\n".join(context_texts)

    # Prompt-Template
    prompt = f"""Du bist ein hilfreicher Assistent. Beantworte die folgende Frage 
ausschließlich auf Basis des gegebenen Kontexts. Wenn die Antwort nicht im Kontext 
steht, sage das klar.

=== KONTEXT ===
{context_string}

=== FRAGE ===
{query}

=== ANTWORT ==="""

    # Pseudo-API-Call (hier würde z.B. openai.chat.completions.create(...) stehen)
    print("\n[PROMPT]\n", prompt[:500], "...\n")  # Nur für Debug-Zwecke
    simulated_answer = "[SIMULIERTE ANTWORT] Das LLM würde hier basierend auf dem Kontext antworten."

    return simulated_answer


# --------------------------------------------------------------------
# Hauptprogramm - alles zusammen
# --------------------------------------------------------------------
def run_rag_pipeline(file_paths: list, query: str) -> str:
    """
    Führt die komplette RAG-Pipeline aus.
    """
    print("=" * 60)
    print("RAG-Pipeline gestartet")
    print("=" * 60)

    # 1. Laden
    docs = load_documents(file_paths)
    if not docs:
        return "[FEHLER] Keine Dokumente geladen."

    # 2. Chunking
    chunks = chunk_documents(docs)

    # 3. Indexieren
    build_index(chunks)

    # 4. Retrieval
    print(f"\n[QUERY] {query}")
    relevant_chunks = retrieve(query, top_k=TOP_K)

    print(f"\n[RETRIEVAL] Top-{TOP_K} Chunks gefunden:")
    for r in relevant_chunks:
        print(f"  - {r['chunk_id']} (Score: {r['score']}) aus '{r['source']}'")

    # 5. Generation
    answer = generate_answer(query, relevant_chunks)
    print(f"\n[ANTWORT]\n{answer}")

    return answer


# --------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Beispiel-Aufruf - passe die Pfade an deine Dokumente an
    example_files = [
        "rag_overview.txt",
        "rag_guide.md",
        "rag_noise.txt"
    ]

    example_query = "Was sind die Vorteile von RAG gegenüber Fine-Tuning?"

    answer = run_rag_pipeline(example_files, example_query)
