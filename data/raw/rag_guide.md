# RAG – Ein technischer Leitfaden für Entwickler

> **Stand:** 2024 | **Niveau:** Fortgeschrittener Entwickler / Master-Student
> Dieses Dokument beschreibt die grundlegenden Konzepte und Bausteine einer RAG-Pipeline.

---

## 1. Was ist ein Large Language Model (LLM)?

Ein **Large Language Model** ist ein neuronales Netz, das auf sehr großen Textmengen trainiert wurde, um Sprache zu verstehen und zu generieren. Die bekanntesten Beispiele sind GPT-4 (OpenAI), Claude (Anthropic) und LLaMA (Meta).

### 1.1 Funktionsweise im Überblick

LLMs werden mit dem Ziel trainiert, das nächste Token in einer Sequenz vorherzusagen. Aus dieser simplen Aufgabe emergiert ein erstaunlich breites Fähigkeitenspektrum:

- Text zusammenfassen
- Code schreiben
- Fragen beantworten
- Übersetzungen erstellen
- Texte klassifizieren

### 1.2 Bekannte Schwächen von LLMs

| Problem | Beschreibung | Schweregrad |
|---|---|---|
| Halluzination | Modell erfindet Fakten, die nicht existieren | Hoch |
| Knowledge Cutoff | Wissen endet am Trainingsende | Mittel |
| Fehlender Domänenkontext | Kein Zugriff auf interne/proprietary Daten | Hoch |
| Kontextfensterbegrenzung | Maximale Eingabelänge limitiert | Mittel |
| Keine Quellenangaben | Keine natürliche Nachvollziehbarkeit | Mittel |

---

## 2. Was ist Retrieval-Augmented Generation (RAG)?

**RAG** ist eine Architektur, die LLMs mit einer externen Wissensbasis kombiniert. Statt das Modell zu fine-tunen, stellt man dem LLM beim Inferenz-Schritt relevante Kontextdokumente bereit.

### 2.1 Grundprinzip

```
Nutzeranfrage → [Retrieval-System] → relevante Chunks → [LLM] → Antwort
```

Die entscheidende Idee: Das **Wissen ist von der Intelligenz getrennt**. Das LLM liefert die sprachliche Intelligenz; die Wissensbasis liefert die Fakten.

### 2.2 Wann macht RAG Sinn?

RAG eignet sich besonders gut, wenn:

- Informationen sich häufig ändern (dynamische Daten)
- Man auf proprietäre oder interne Dokumente zugreifen will
- Nachvollziehbarkeit (Quellen) wichtig ist
- Fine-Tuning zu teuer oder aufwändig wäre

---

## 3. Die RAG-Pipeline im Detail

Eine typische RAG-Pipeline besteht aus zwei Phasen: **Offline-Ingestion** und **Online-Retrieval + Generation**.

### 3.1 Phase 1: Ingestion (Offline)

#### 3.1.1 Dokumente laden

Die Wissensbasis kann aus verschiedenen Quellen bestehen:

- PDF-Dateien
- Markdown-Dokumente
- Websites (gecrawlt)
- Datenbanken
- APIs

#### 3.1.2 Chunking

Dokumente werden in kleinere Einheiten (Chunks) aufgeteilt. Gute Chunking-Strategien sind entscheidend für die spätere Retrieval-Qualität.

**Gängige Chunking-Strategien:**

| Strategie | Beschreibung | Typische Chunk-Größe |
|---|---|---|
| Fixed-Size | Einfaches Aufteilen nach Token-Anzahl | 256–512 Tokens |
| Sentence-based | Splitting an Satzgrenzen | variabel |
| Semantic | Splitting nach semantischen Einheiten | variabel |
| Recursive | Hierarchisches Splitting mit Overlap | 256–1024 Tokens |

> **Wichtig:** Ein Overlap (Überlappung) zwischen Chunks verhindert, dass Informationen an Chunk-Grenzen verloren gehen.

#### 3.1.3 Embedding

Jeder Chunk wird durch ein **Embedding-Modell** in einen dichten Vektor umgewandelt. Dieser Vektor repräsentiert die semantische Bedeutung des Textes im hochdimensionalen Raum.

Populäre Embedding-Modelle:

- `text-embedding-3-small` / `text-embedding-3-large` (OpenAI)
- `all-MiniLM-L6-v2` (Sentence-Transformers, Open Source)
- `e5-large-v2` (Microsoft)
- `jina-embeddings-v2` (Jina AI)

#### 3.1.4 Vektordatenbank

Die erzeugten Vektoren werden zusammen mit Metadaten in einer **Vektordatenbank** gespeichert.

Bekannte Vektordatenbanken:
- **Chroma** – leichtgewichtig, gut für Prototypen
- **Weaviate** – produktionsreif, Cloud-native
- **Qdrant** – hohe Performance, in Rust geschrieben
- **Pinecone** – fully managed, einfache API
- **FAISS** – Facebook-Bibliothek, kein Server nötig

---

### 3.2 Phase 2: Retrieval + Generation (Online)

Wenn ein Nutzer eine Frage stellt, läuft folgendes ab:

1. **Query Encoding:** Die Anfrage wird mit demselben Embedding-Modell in einen Vektor umgewandelt
2. **Similarity Search:** In der Vektordatenbank werden die k ähnlichsten Chunks gefunden (meistens per Cosine Similarity oder Dot Product)
3. **Context Assembly:** Die gefundenen Chunks werden zu einem Kontext-String zusammengestellt
4. **Prompt Construction:** Ein Prompt wird gebaut: `[Systemanweisung] + [Kontext] + [Nutzerfrage]`
5. **LLM-Inferenz:** Das LLM generiert die Antwort auf Basis des angereicherten Prompts

#### 3.2.1 Retrieval-Strategien

```
Standard-Retrieval:   Query → Top-k Chunks → LLM
HyDE:                 Query → LLM (hypothetical answer) → Embedding → Top-k Chunks → LLM
Re-Ranking:           Query → Top-n Chunks → Re-Ranker → Top-k Chunks → LLM
Multi-Query:          Query → mehrere Sub-Queries → jeweils Top-k → dedupliziert → LLM
```

---

## 4. Vor- und Nachteile von RAG

### 4.1 Vorteile

- ✅ **Aktualität:** Wissensbasis kann jederzeit aktualisiert werden
- ✅ **Transparenz:** Quellendokumente können referenziert werden
- ✅ **Kein Retraining:** Modell bleibt unverändert
- ✅ **Skalierbarkeit:** Wissensbasis kann beliebig wachsen
- ✅ **Datenschutz:** Sensible Daten verlassen die eigene Infrastruktur nicht
- ✅ **Kostengünstig:** Kein teures Fine-Tuning notwendig

### 4.2 Nachteile

- ❌ **Retrieval-Qualität kritisch:** Schlechtes Retrieval → schlechte Antworten
- ❌ **Latenz:** Zusätzlicher Retrieval-Schritt erhöht die Antwortzeit
- ❌ **Chunk-Design ist Kunst:** Gutes Chunking erfordert Erfahrung und Experimente
- ❌ **Out-of-Context-Probleme:** Wenn nötiger Kontext auf mehrere Chunks verteilt ist
- ❌ **Mehrsprachigkeit:** Embedding-Modelle performen nicht immer crosslingual optimal
- ❌ **Infrastruktur-Overhead:** Vektordatenbank muss betrieben und gepflegt werden

---

## 5. Evaluation einer RAG-Pipeline

Eine RAG-Pipeline zu bauen ist eine Sache – sie zu evaluieren eine andere.

### 5.1 Wichtige Metriken

| Metrik | Beschreibung |
|---|---|
| **Retrieval Recall** | Wurden die relevanten Dokumente gefunden? |
| **Retrieval Precision** | Sind die gefundenen Dokumente tatsächlich relevant? |
| **Answer Faithfulness** | Ist die Antwort durch den Kontext gedeckt? |
| **Answer Relevance** | Beantwortet die Antwort die eigentliche Frage? |
| **Hallucination Rate** | Wie oft erfindet das Modell Informationen? |

### 5.2 Evaluation-Frameworks

- **RAGAS** – spezialisiertes Framework für RAG-Evaluation
- **TruLens** – allgemeines LLM-Evaluationstool
- **LangChain Evaluators** – gut integriert in LangChain-Pipelines

---

## 6. Weiterführende Ressourcen

- Lewis et al. (2020): *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Originalpaper)
- LangChain Dokumentation: https://docs.langchain.com
- LlamaIndex Dokumentation: https://docs.llamaindex.ai
- Weaviate Blog: Best Practices für Produktions-RAG-Systeme

---

*Letzte Überarbeitung: Dezember 2024*
