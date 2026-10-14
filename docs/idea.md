# `auth-rag`
## Introduzione
L'idea del progetto è quella di creare una libreria interna che permetta di integrare la Retrieval Augmented Generation (**RAG**), su degli applicativi che sfruttano i Large Language Models (**LLM**). La peculiarità del progetto consiste nell'integrazione esplicita di un layer di autorizzazione che regola il controllo degli accessi basato sugli attributi (**ABAC**).

La libreria permette, in sostanza, di comporre due tipi di pipeline: ingestione documenti per creare una base di conoscenza (`ingestion`) e ricerca sui dati interni per generare una risposta (`query`).
Ciascuna pipeline è configurabile con una serie di strumenti componibili - utilizzabili in diversi contesti, dalle Proof-Of-Concept, passando per lo sviluppo locale, fino ad ambienti di produzione distribuiti - che permettono di estendere la conoscenza di un determinato modello su dei dati interni, mantenendo comunque le caratteristiche granulari di accesso.

### Stato dell'arte
Esistono attualmente diversi framework per implementare RAG per applicativi basati su LLM, come LlamaIndex, LangChain e LangGraph. Questi strumenti offrono già una buona astrazione per la gestione dei flussi di lavoro, l'ingestione dei documenti, lo splitting, l'embedding e l'interfacciamento con i vector database. Tra questi framework vi sono, in tutti i casi, strumenti e scelte implementative differenti, che li rendono adatti a scopi diversi.

La generalità di questi strumenti è un'arma a doppio taglio, in quanto ogni qual volta si implementa un RAG è necessario realizzare una specifica pipeline ad hoc da zero. Inoltre, nella loro generalità ciascun framework può possedere caratteristiche che comunque gli altri non hanno.

Poi da un punto di vista computazionale la generalità di questi strumenti è proporzionale al peso delle dipendenze che si portano dietro, rendendoli poco adatti in ambienti di produzione.

Ma soprattutto, tutti i framework per RAG presentano un limite strutturale comune: mancano di un supporto nativo, granulare e standardizzato per il controllo degli accessi ai dati. Nella maggior parte delle architetture RAG commerciali o open-source attuali, la sicurezza viene gestita a livello perimetrale (autenticazione all'applicazione), tramite filtri grossolani sui metadati delle query (es. filtrando per certi campi) oppure affidandosi a motori esterni di sicurezza.
Questo approccio mostra forti limiti in contesti enterprise perché manca di granularità, cioè non supporta logiche dinamiche complesse in cui l'accesso a un documento può dipendere da una combinazione di attributi dell'utente, del documento stesso e del contesto ambientale. Ciò determina un elevato livello di accoppiamento, poiché la logica di autorizzazione risulta intrinseca al codice della pipeline RAG: i processi di manutenzione, auditing e scalabilità del sistema subiscono significativi rallentamenti.

### Motivazione
Il progetto si colloca in questo scenario come una soluzione specializzata che fornisce un alto livello di astrazione senza però limitare le possibilità di configurabilità del RAG, dove l'obiettivo è offrire una struttura fissa che raccolga le funzionalità essenziali e le migliori scelte implementative dei framework esistenti. Non mira a sostituire i framework di orchestrazione attuali, ma ad offrire principalmente una standardizzazione sugli strumenti e a colmare il loro gap di sicurezza introducendo l'ABAC all'interno delle pipeline di ingestion e query.

In sostanza il progetto risolve tali problemi offrendo un solo facade configurabile, semplificando l'integrazione e la manutenibilità nei progetti, ma imponendo un middleware di autorizzazione.


## Architettura
L'architettura della libreria `auth-rag` si configura nel pattern di architettura esagonale, dove il core definisce il flusso di dati esponendo all'esterno delle interfacce configurabili attraverso la dichiarazione degli adattatori che eseguono uno specifico compito. In generale si distinguono due parti, corrispondenti alle due tipologie di pipeline che si possono comporre: **ingestion** e **query**.
Si noti che per quanto le due pipeline siano separate risultano comunque accoppiate da alcuni strumenti interni, come i modelli di embedding, gli store selezionati e non di meno il modello di **autorizzazione**. 
Per tale motivo è necessario che le pipeline di ingestion e query siano opportunamente cablate da un file di **configurazione** che assegna i relativi strumenti condivisi e non.

### Ingestion
La pipeline di ingestion gestisce l'esecuzione in serie dei seguenti strumenti:

1. `Loader`: carica i documenti da fonti interne o esterne e li converte in formato Markdown chiamando un `Converter`:
    - `LocalLoader`: caricatore di un documento interno fornito un percorso nel file system;
    - `RemoteLoader`: caricatore di documenti tramite chiamate HTTP REST fornito un endpoint.
2. `Labeler`: scrive eventualmente gli attributi di accesso sui documenti basandosi sul principio che etichettare non è decidere: si marca il dato per *cosa è*, mai per *chi lo vede*.
Si noti che un attributo invalido o mancante ferma la funzione di ingestione invece di far saltare il documento. Sono presenti le seguenti versioni:
    - `PropagatingLabeler`: valida gli attributi già arrivati col documento, quando c'è un produttore a monte;
    - `ManifestLabeler`: li legge da un manifesto JSON per id documento, per corpus curati a mano;
    - `StaticLabeler`: applica le stesse costanti a tutto ciò che quella pipeline ingerisce.
3. `Chunker`: divide il documento in parti atomiche dette chunk, ereditando i metadati del documento di origine, inclusi eventuali attributi d'autorizzazione, per cui l'enforcement avrà un solo posto da cui leggere. In funzione del criterio di taglio distinguiamo:
    - `FixedSizeChunker`: divide il testo in finestre di caratteri a lunghezza fissa;
    - `SentenceChunker`: raggruppa frasi in chunk utilizzando un tokenizzatore specifico;
    - `HierarchicalChunker`: divide il markdown in chunk seguendo la gerarchia delle sezioni;
    - `RecursiveCharacterChunker`: divide il testo con i separatori fino ad ottenere la dimensione configurata;
    - `SemanticChunker`: divide il testo in insiemi di frasi coerenti usando il machine learning.
4. `Store`: collezione che contiene i veri e propri chunk. Sono presenti tre possibili tier in base alle dimensioni del progetto e al punto di allocazione della struttura dati:
    - `VolatileStore`: dizionario in memoria, non durevole;
    - `PersistentStore`: file SQLite singolo, durevole su un solo processo;
    - `RemoteStore`: tabella Postgres, durevole e condivisa fra processi.
5. `Index`: indice dei chunk, non li contiene ma decreta il modo per ritrovarli. In altre parole lo `Store` è la verità su *cos'è* un chunk, l'`Index` su *come si ritrova*: un index conserva solo gli id più la propria rappresentazione e risolve i chunk dallo store condiviso, così il testo esiste in una copia sola.  
Anche `Index` esiste nei tre tier `Volatile`, `Persistent` e `Remote`, che differiscono solo nella tecnologia sottostante e nel tipo di allocazione ma mai nel comportamento esterno: il nome dichiara la garanzia offerta, non il motore che la realizza. La pipeline permette di gestire nativamente più indici di diversa tipologia:
    - `LexicalIndex`: vettori sparsi BM25, per la corrispondenza esatta dei termini tramite l'algoritmo TF-IDF: eccelle sulla corrispondenza esatta e non riconosce un sinonimo;
    - `SemanticIndex`: vettori densi, i.e. embedding ottenuti chiamando un `Embedder`, per la vicinanza di significato, calcolato con una metrica di distanza (coseno dell'angolo): trova ciò che significa la stessa cosa scritto con altre parole ma annega termini rari, codici e nomi propri;
    - `GraphIndex`: grafo di relazioni fra concetti, ottenute tramite un `Extractor`, per la vicinanza di contesto: valuta un chunk per i legami che possiede, e paga dove due chunk non sono simili ovvero dove la similarità non li avrebbe mai messi insieme.

Completano l'ingestione gli strumenti che non sono stadi a sé, ma parti iniettate negli altri:
- `Converter`: usato dal loader, converte il payload in markdown scegliendo il parser dal media type. Un formato non riconosciuto solleva, mai un fallback silenzioso;
- `Cleaner`: normalizzazione testuale, a disposizione e non cablata — normalizzare a monte distruggerebbe la struttura markdown su cui lavora l'`HierarchicalChunker`;
- `Embedder`: testo in vettori densi, iniettato nel `SemanticIndex`; locale o servito da un'API;
- `Extractor`: chunk in relazioni fra concetti, iniettato nel `GraphIndex` come l'embedder nel semantico.


### Query
La pipeline di query, che in realtà corrisponde alla Retrieval Augmented Generation, gestisce l'esecuzione in serie dei seguenti stadi:

1. `Index`: ogni index recupera (**Retrieval**) i propri candidati per la stessa query e vi applica il filtro di autorizzazione: gli indici utilizzati devono essere al più gli stessi della fase di ingestion.
2. `FusionRanker`: fonde le liste dei diversi index in un'unica classifica: l'idea è che la corroborazione fra metodi diversi è un segnale più affidabile di un punteggio alto prodotto da uno solo. Sono distinti qui in due famiglie, una basata sul rank e l'altra sugli score:
    - `RankFusionRanker`: fonde per rango, partendo solo dalle posizioni premia i chunk trovati da più retriever indipendenti:
        - `ReciprocalRankFusionRanker`: fonde leggendo solo le posizioni, ignorando i punteggi;
    - `ScoreFusionRanker`: fonde i punteggi ottenuti per chunk:
        - `RelativeScoreFusionRanker`: normalizza sul minimo e massimo di ogni lista.
        - `DistributionScoreFusionRanker`: normalizza su media e deviazione standard di ogni lista.
3. `Expander` (opzionale): un grafo che entra dai chunk già fusi invece che dalla query, e aggiunge quelli a cui portano i loro concetti. Ognuno eredita il punteggio del chunk da cui è stato raggiunto, scalato per la distanza, ed è filtrato come ogni risultato di un index; sta dopo la fusione perché la sua lista deriverebbe da quella degli index, e la fusione conterebbe due volte lo stesso segnale.
4. `Reranker`: riordina i candidati sopravvissuti, confrontandoli con la query:
    - `CrossReranker`: cross-encoder che valuta congiuntamente la coppia (query, chunk).
5. `Augmenter`: assembla il prompt da istruzioni, domanda e contesto (**Augmenter**):
    - `PromptAugmenter`: un messaggio di sistema e uno utente, con le fonti numerate e attribuite.
6. `LLMClient`: invia il prompt al modello e ne restituisce la risposta (**Generation**), completa oppure in streaming. I client implementati sono:
    - `OllamaClient`: modelli serviti localmente da Ollama;
    - `OpenAIClient`: modelli serviti dall'API OpenAI.


### Autorizzazione
Il layer è trasversale alle due pipeline, e parte da una scelta di confine: la libreria non vede mai l'identità di chi interroga. Chi decide sta a monte ed è poi la libreria che fornisce il vocabolario, il linguaggio del predicato, e la garanzia che il predicato venga applicato.
Gli elementi fondamentali del layer di autorizzazione sono: 
- `AccessSchema`: il vocabolario chiuso degli attributi che un deployment dichiara: nome, tipo, se multi-valore, se obbligatorio. Sta in un file e non nel codice perché ingestione e query possono essere processi distinti e devono leggere la stessa dichiarazione.
- `Filter`: l'algebra dei predicati, neutra sia verso il motore di policy sia verso il backend di ricerca. Offre i seguenti predicati: corrispondenza di un attributo con un insieme di valori, congiunzione, disgiunzione, negazione, e una costante che dichiara esplicitamente l'assenza di restrizioni.

Lo schema ha tre consumatori: valida ciò che il labeler scrive, dice agli index cosa indicizzare, rifiuta un predicato che nomini un attributo mai dichiarato. La sua presenza è ciò che distingue un deployment: senza schema non c'è ABAC, con uno schema etichettatura e filtro diventano entrambi obbligatori.


Si noti che l'enforcement vive nell'index, non nella pipeline: gli index sono API pubblica e qualcuno li userà senza pipeline, quindi un controllo di sicurezza non può dipendere solo da quale wrapper è stato scelto.

Il processo di filtraggio avviene prima della fusione, non dopo. Non per preferenza fra pre- e post-filtering, ma perché il riferimento è il ranker: la fusione per rango legge le posizioni dentro ogni lista, quella per punteggio normalizza sui suoi estremi. Lasciare dentro candidati non autorizzati significherebbe che quei candidati spostano il rango degli altri e fissano i limiti di normalizzazione — cioè che il punteggio di ciò che si vede dipenderebbe da ciò che non si può vedere.

Schema e filtro si richiedono nei due sensi, e ogni disaccordo solleva invece di essere ignorato: schema senza filtro, filtro senza schema, o chunk etichettati che arrivano a un index che non applica niente. Sono tutti errori di cablaggio, non richieste malformate.


### Configurazione
Le due pipeline di query ed ingestion, incluso il livello di autorizzazione, devono essere opportunamente accoppiate affinché condividano una serie di strumenti.
La configurazione base è tipizzata da variabili d'ambiente, per tutte le chiavi tranne per i percorsi della dichiarazione di accesso: l'assenza significa nessun ABAC. Il cablaggio sta in un unico composition root, l'unico punto che conosce l'ambiente, sotto forma di codice (l'alternativa è una factory, dove si dichiara in un file un nome che con un registro di stringhe viene risolto a runtime). La facade `RagPipeline` costruisce le due pipeline condividendo store e index, ed è questa condivisione a garantire che ciò che l'ingestione scrive sia ciò che la query legge. Tale facade accoppia esplicitamente le pipeline, vincolandole a stare in un unico processo.
Le due pipeline restano però costruibili anche separatamente, senza la facade: un processo cabla la sola ingestione e un altro la sola query, purché condividano gli stessi tier remoti, oltre che la stessa dichiarazione di accesso e lo stesso modello di embedding. In quel caso la garanzia che la query legga ciò che l'ingestione ha scritto non viene più dalla condivisione degli oggetti in memoria, ma dal fatto che i due composition root puntano allo stesso backend.


## Glossario
- `Source`: identifica la sorgente dei documenti tramite id, nome, origine, data e attributi di accesso.
- `Document`: formato del documento con testo e sorgente.
- `Metadata`: metadati del chunk con titolo e sorgente.
- `Chunk`: unità atomica di testo con relativo id, testo e metadati.
- `Message`: testo da inviare ad un LLM comprendente system prompt, query e contesto derivante dai chunk.
- `RemoteDocument`: payload grezzo consegnato da un servizio esterno, con byte, media type e attributi di accesso.
- `ScoredChunk`: chunk accompagnato dal punteggio assegnatogli da un index o da un ranker.
- `Triple`: relazione fra due concetti come l'estrattore l'ha letta, con soggetto, predicato e oggetto.
- `Attribute`: singolo attributo di accesso dichiarato, con nome, tipo, molteplicità e obbligatorietà.
- `AccessSchema`: il vocabolario chiuso degli attributi di accesso dichiarati da un deployment.
- `Filter`: predicato di autorizzazione neutro, applicato dagli index e deciso a monte della libreria.