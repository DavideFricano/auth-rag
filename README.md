# Auth RAG

A unified, in-process interface for Retrieval Augmented Generation, wrapping the relevant tools for
LLM applications from prototyping to production — with **attribute-based access control enforced at
retrieval time**, so a corpus that not every user may read in full can still be served by a single
pipeline.

Every stage is a port with interchangeable implementations, wired in one place. Nothing is a
service: it is a library you compose.

## Motivation — access control belongs in retrieval

The moment a RAG system is pointed at a real corpus — clinical records, tenants sharing an index,
anything with a classification — the question stops being *what is relevant* and becomes *what is
this requester allowed to see*. Most pipelines answer it in the wrong place, and three things go
wrong:

- **The LLM is treated as the security boundary.** It is not one. A model that receives a document
  it should not have received has already been given it; whether it then declines to quote it is a
  matter of instruction-following, not of access control.
- **Filtering happens after ranking.** Post-filtering breaks `top_k` — you ask for ten results and
  get four — and it leaks *existence*: the gaps tell you something is there.
- **Filtering happens after fusion.** Subtler, and the reason this project puts the check where it
  does. Reciprocal Rank Fusion reads *positions* within each retriever's list, and score-based
  fusion normalizes over each list's *minimum and maximum*. Leave unauthorized chunks in those
  lists and they shift the rank of the authorized ones and set the normalization bounds — the score
  of what you see depends on what you cannot see.

So the enforcement point is the **index**, before fusion: candidates are resolved, then the
predicate is applied, and only authorized chunks ever reach a ranker.

## How access control works

**One declaration.** A deployment declares a closed vocabulary of access attributes in a JSON file
(`AccessSchema`), versioned and reviewed like code. Its *presence is what distinguishes a deployment
with access control from one without* — unset, nothing labels and nothing filters.

```json
[
  { "name": "tenant",         "type": "keyword", "required": true },
  { "name": "classification", "type": "keyword", "required": true },
  { "name": "care_team",      "type": "keyword", "multi": true }
]
```

One declaration, three consumers: it validates what ingestion writes, it rejects a query predicate
naming an attribute nobody declared, and it tells the retrieval side which chunks count as labelled.

**At ingestion, labelling is not deciding.** A `Labeler` runs between loading and chunking and marks
each document for *what it is* — never for *who may see it*. Attributes either arrive with the
document (a gateway upstream knows the patient, the classification, the care team) or are curated in
a manifest. Because the rules live elsewhere, **changing a policy never forces a re-ingestion**.

**At query time, a neutral predicate.** The caller hands in a `Filter` — a small closed algebra
(`Match`, `And`, `Or`, `Not`, `Allow`) that is neither the policy engine's dialect nor the storage
backend's. It travels through the pipeline uninterpreted and is applied by every index.

```python
results = pipeline.retrieve(
    "toxicity management after immunotherapy",
    filter=And(clauses=[
        Match(attribute="tenant", values={"acme"}),
        Match(attribute="classification", values={"public", "internal"}),
    ]),
)
```

Three properties worth stating plainly:

- **Schema and filter are inseparable, in both directions.** With a schema declared, omitting the
  filter raises — `Allow()` is how a call states it has no restriction, distinct from having
  forgotten the argument. Without a schema, passing a filter raises too: there is no vocabulary to
  validate it against, so a negated predicate would silently admit an unlabelled chunk.
- **A chunk missing a required attribute is denied before the predicate runs**, `Allow()` included.
  Default-deny is a property of the enforcement point, not of how the policy happens to be written.
- **The check is in Python, in the index base class**, so it holds for *every* backend and cannot be
  omitted by the author of a new index. Pushing the predicate down into a backend is an optimization
  for recall, never the condition for the filter to exist.

**What stays outside.** The library never sees an identity. Compiling `(subject, action, environment)`
into a filter is the job of a Policy Enforcement Point sitting in front of it, backed by an external
XACML Policy Decision Point — a separate process, by design. See
[docs/roadmap/sources-abac.md](docs/roadmap/sources-abac.md).

## The rest of the pipeline

Two lifecycles that share a store and a set of indexes: **ingestion** (offline) and **query**
(online).

| stage | what ships |
|---|---|
| **Loading** | local filesystem, or HTTP pull from an external service (a FHIR gateway lives outside the library and delivers neutral bytes) |
| **Conversion** | dispatch on media type: Docling for PDF/DOCX/PPTX/images, MarkItDown for CSV/JSON/XLSX/HTML, decode for text |
| **Chunking** | five strategies — markdown-hierarchical (default), sentence, recursive-character, fixed-size, semantic |
| **Storage** | the chunk data lives once in a shared store; indexes hold only ids plus their own representation |
| **Retrieval** | hybrid — dense (cosine) and sparse (BM25 with corpus IDF computed server-side) |
| **Fusion** | RRF, or score fusion with min-max / distribution normalization |
| **Reranking** | cross-encoder over the fused candidates |
| **Generation** | Ollama or OpenAI, fed a neutral `list[Message]` |

Every component exists in three tiers — **volatile**, **persistent**, **remote** — named for the
guarantee they give (does it survive a restart? is it shared across processes?) rather than for the
product underneath.

## Graph-based context expansion

*Planned; nothing of this is implemented yet.* The original thesis of the project, kept because the
retrieval architecture is already shaped to receive it: a relation index plugs in as one more index
alongside the similarity ones, with no change to the query pipeline.

In canonical Graph RAG the graph **is** the index — an extractor builds a knowledge base where every
node is a specific embedding, and the query navigates it. That demands extraction accurate enough to
produce informative, non-redundant relations: domain experts and a high construction cost.

Here the roles are inverted. The **vector store remains the main index** and the graph sits on top of
it solely to expand context. Nodes are *subconcepts* rather than embeddings, so several of them can
resolve to the same chunk, giving one vector multiple relational entry points. Edges connect chunks
linked by an explicit relation — and the gain is precisely where those chunks are *not* semantically
similar, because similarity alone would never have put them together. Since the graph only has to
record that two nodes in different chunks are related, and not to be a faithful map of the domain, it
stays coarse, cheap to build and cheap to maintain.

The open questions — how a traversal becomes a score, where the seeds come from, how access control
interacts with reachability, and whether the expansion pays for itself at all — are written up in
[docs/roadmap/graph.md](docs/roadmap/graph.md).

## Installation

This project uses [uv](https://docs.astral.sh/uv/). A single command installs everything:

```bash
uv sync
```

## Configuration

When building your own entry point, `auth_rag.config.Settings` reads typed settings from the
environment or a `.env` file. Every value has a local default, so an empty environment works out of
the box — see `Settings` for the current fields, and `.env.example` for the shape.

Two fields deliberately have no default: `access_schema_path` and `access_manifest_path`. The library
has no opinion on where a deployment keeps its access declaration, and their absence *means*
something — no schema, no access control.

## Documentation

| document | scope |
|---|---|
| [docs/architecture.md](docs/architecture.md) | the whole system: layers, types, the two pipelines, and why the boundaries are where they are |
| [docs/roadmap/sources-abac.md](docs/roadmap/sources-abac.md) | more input sources, and the access-control layer end to end (PEP, PDP, obligations) |
| [docs/roadmap/graph.md](docs/roadmap/graph.md) | the relation graph: decisions taken, decisions open, release sequence |
