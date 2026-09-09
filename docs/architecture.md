# Autograph RAG — architettura

Due pipeline, cicli di vita separati: **Ingestion** (offline, popola lo store) e **Query** (online, risponde). Sugli archi è indicato l'oggetto che passa da un blocco al successivo.

## Ingestion pipeline (offline)

```mermaid
flowchart LR
    subgraph SRC[" Sources "]
        direction TB
        srcFs[/" FileSystem<br>(local dir) "/]
        srcGw[/" Gateway<br>(HTTP, fronts FHIR) "/]
        srcFs ~~~ srcGw
    end

    subgraph LOAD[" Loader "]
        direction TB
        ldFs[" FileLoader "]
        ldApi[" ApiLoader<br>(pull) "]
        ldFs ~~~ ldApi
    end

    subgraph CONV[" Converter "]
        direction TB
        cvDocling[" Docling<br>(PDF/DOCX/PPTX/img) "]
        cvMarkit[" MarkItDown<br>(CSV/JSON/XLSX/HTML) "]
        cvText[" decode<br>(text/*) "]
        cvDocling ~~~ cvMarkit ~~~ cvText
    end

    srcFs -- path --> ldFs
    srcGw -- RemoteDocument --> ldApi
    LOAD -- "convert_file / convert_stream" --> CONV
    CONV -- Document --> C[" Chunker<br>(Hierarchical/Semantic/<br>Sentence/Fixed/Recursive) "]

    %% dato dei chunk: scritto una volta sola nello store condiviso
    C -- list[Chunk] --> S[" Store<br>(Volatile/Persistent/Remote) "]
    S -- chunks --> DBc[( chunks )]

    %% indici: solo id + rappresentazione, risolvono i chunk dallo store
    C -- list[Chunk] --> E[" Embedder<br>(Local/OpenAI) "]
    E -- NDArray[float32] --> IV[" SemanticIndex<br>(Qdrant, cosine) "]
    IV -- "id + dense vector" --> DBv[( points )]
    C -- list[Chunk] --> IL[" LexicalIndex<br>(Qdrant, BM25/IDF) "]
    IL -- "id + sparse vector" --> DBs[( points )]
    IV -.->|"store.get(ids)"| S
    IL -.->|"store.get(ids)"| S

    srcFs:::io
    srcGw:::io
    ldFs:::step
    ldApi:::step
    cvDocling:::step
    cvMarkit:::step
    cvText:::step
    C:::step
    E:::step
    S:::store
    IV:::store
    IL:::store
    DBv:::db
    DBs:::db
    DBc:::db
    SRC:::group
    LOAD:::group
    CONV:::group

    classDef io fill:#eceff1,stroke:#607d8b,color:#263238
    classDef step fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef store fill:#fff3e0,stroke:#ef6c00,color:#e65100
    classDef db fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    classDef group fill:#fafafa,stroke:#bdbdbd,color:#424242
```

## Query pipeline (online)

```mermaid
flowchart LR
    q[/"query"/] -->|"str"| BM["LexicalIndex<br>(Qdrant, BM25/IDF)"] & VR["SemanticIndex<br>(Qdrant, cosine)"] & GR["GraphIndex<br>(previsto)"]
    q -- str --> AUG["PromptAugmenter"]
    VR -- list[ScoredChunk] --> F["FusionRanker<br>(RRF/RSF/DBSF)"]
    BM -- list[ScoredChunk] --> F
    GR -- list[ScoredChunk] --> F
    F -- list[ScoredChunk]<br> --> RR["Reranker<br>(CrossEncoder)"]
    RR -- list[ScoredChunk]<br> --> AUG
    AUG -- list[Message] --> LLM["LLMClient<br>(Ollama/OpenAI)"]
    LLM -- str --> ANS[/"answer"/]
    n1[/"system_prompt"/] -- str --> AUG

     q:::io
     BM:::step
     VR:::step
     GR:::step
     AUG:::step
     F:::rank
     RR:::rank
     LLM:::step
     ANS:::io
     n1:::io
    classDef io fill:#eceff1,stroke:#607d8b,color:#263238
    classDef step fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef rank fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef db fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
```

## Tipi che scorrono

| Tipo | Campi |
|---|---|
| `Source` | `id` (nome file o `external_id`), `name`, `origin`, `time`, `access` (attributi ABAC, vuoto senza schema) |
| `Document` | `text`, `source: Source` |
| `Metadata` | `source: Source`, `title`, `page?` |
| `Chunk` | `id` (`source.id` + content_hash), `text`, `metadata: Metadata` |
| `ScoredChunk` | `chunk: Chunk`, `score: float` |
| `Message` | `role` (system/user/assistant), `content` |

## Estensioni

Questo documento descrive il **base**: le due pipeline e i tipi che scorrono. Ogni estensione
ha il suo documento, che parte da qui e non lo ridisegna.

| documento | asse |
|---|---|
| [roadmap/sources-abac.md](roadmap/sources-abac.md) | più sorgenti in ingresso, e il controllo accessi ABAC sul retrieval |
| [roadmap/graph.md](roadmap/graph.md) | il grafo di relazioni che allarga il contesto recuperato |
