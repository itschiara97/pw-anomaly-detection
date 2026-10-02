# Verifica paper, implementazioni e risultati

Stato: revisione delle discrepanze, verifica dei quattro risultati archiviati, controllo delle fonti citate e aggiornamento del testo completati il 2 ottobre 2026. Su richiesta dell'utente, nell'elaborato il sistema operativo è indicato soltanto come Linux e la sua versione specifica non è considerata nel confronto. Non si dichiara una nuova certificazione delle esecuzioni. Le cause non isolate sperimentalmente restano esplicitamente ipotesi o sviluppi futuri.

## Risultati verificati

I rapporti `verifiche/metriche_<modello>.json` registrano il ricalcolo su tutte le 124 mappe 1024×1024, senza eseguire i modelli. Per ciascuno dei quattro metodi sono verificati 16 valori: tre aree principali, F1 immagini/pixel, precisione e richiamo immagini/pixel, soglie immagini/pixel, FPR@TPR95 ed estremi delle scale immagine/mappa. Differenza massima osservata: **0**. Il confronto è tra pipeline complete, non a compute uguale.

Comando riproducibile dalla radice della workspace:

```sh
repository_pw/.venv/bin/python PW_Anomaly_Detection/scripts/verifica_risultati_archiviati.py
```

La verifica controlla anche configurazioni, sorgenti, checkpoint, dataset, ground truth, catena dei manifest e mappe tramite le impronte registrate. Non sovrascrive metriche o manifest originali. Il confronto nell'elaborato si basa sui dati, sul protocollo comune e sull'integrità degli artefatti; non richiede identità della versione specifica di Linux. I dettagli tecnici dei controlli automatici restano nei JSON, senza alterare il gate o dichiararlo superato.

Sono inoltre passati i 16 test esistenti eseguiti con:

```sh
repository_pw/.venv/bin/python -m pytest repository_pw/PaDiM/tests/test_comparability.py repository_pw/FastFlow/tests/test_fastflow.py -q
```

Questi test coprono protocollo, concatenazione delle feature PaDiM, formule di loss/mappa FastFlow, forma dei blocchi e provenance. Non provano da soli equivalenza dei checkpoint o risultati delle pubblicazioni.

## CCW-SSIM: evidenze e significato

Sorgenti effettive: `Training.py:color_cwssim_loss`, `ColorUtils.py:prepare_image_colorssim`, `Prediction.py:compute_anomaly_map`, `Steerables/AnomalyMetrics.py:cw_ssimcolor_metric`, `Steerables/metrics_TF.py:cw_ssim_score`.

1. Il preprocessing usa conversione RGB→XYZ, canali opponenti, filtri spaziali e ritorno a Lab, seguito da divisione per 102. Non è semplicemente una conversione RGB→Lab.
2. Il paper di Berasi, equazioni 8–11, usa distanza cromatica euclidea e combinazione moltiplicativa con CW-SSIM. Il training locale usa MSE sui canali a/b normalizzati, mediata globalmente e divisa per 45. La mappa usa MSE mediata 7×7 e moltiplicata per 10. Il rapporto dei coefficienti è 450; le medie sono diverse, quindi non si afferma che ogni contributo effettivo sia esattamente 450 volte maggiore. È una variante documentabile, non identità della metrica pubblicata.
3. La perdita è cromatico-strutturale nel percorso eseguito, ma il contributo cromatico al suo valore è piccolo nei campioni diagnostici. Non sono stati misurati gradienti né effetti causali sulla prestazione.
4. `InvertFirstLabAxis=True` è attivo; l'affermazione secondo cui il decoder richiederebbe luminanza con segno invertito non è sostenuta dal checkpoint testato. La formulazione è stata corretta nella tesi.
5. La formula CW-SSIM usa il modulo del prodotto dei coefficienti e conserva il punteggio quando un segnale è negato globalmente. La componente cromatica esclude L. L'inversione di L non è quindi una spiegazione del divario delle metriche raw. Non è stata rimossa dalla configurazione storica.
6. I nove falsi positivi alla soglia globale di massimo F1 sono accertati. Per otto il massimo è a 1–3 pixel dal bordo; per `good/030.png` la distanza è 312 pixel. La risposta periferica è un indizio concreto, non una dimostrazione di quale parte della pipeline la generi. Non è stata introdotta una correzione test-tuned dei bordi.
7. Il valore 0,982 della fonte riguarda una ROC normalizzata fino a FPR 0,3, non l'AUROC completa né AU-PRO locale. La fonte sintetica non specifica tutti i dettagli del valutatore: non si presume una particolare formula di standardizzazione della ROC parziale.

### Diagnostica autorizzata

L'utente ha autorizzato soltanto un test del checkpoint su poche patch normali. Il test ha usato CPU, dodici patch 128×128, le prime tre immagini `test/good`, quattro coordinate fissate per immagine. Non ha riaddestrato o prodotto mappe ufficiali.

Evidenza: `verifiche/diagnostica_ccw_checkpoint.json`, con hash del checkpoint, immagini e script. RMSE L originale: 2,08–7,03 unità Lab; con inversione: 63,90–72,02. Perdita strutturale raw/invertita identica: 0,15351889796. Perdita totale: 0,15352732645; differenza: 8,42849e-6. La portata resta limitata alle dodici patch: non è un'ablazione né una nuova misura di generalizzazione.

```sh
repository_pw/.venv/bin/python PW_Anomaly_Detection/scripts/diagnostica_ccw_checkpoint.py
```

La cronologia CSV ha 200 righe e ultimo indice 199. Il percorso è quello configurato, ma il manifest storico non registra l'hash del CSV: non si presenta la cronologia come prova crittograficamente concatenata al checkpoint. Il rapporto del ricalcolo registra ora l'hash corrente e tale limite.

## Altri metodi

- **PaDiM:** backbone congelato, tre livelli, selezione di 550 canali (non proiezione Johnson–Lindenstrauss), gaussiane locali regolarizzate e distanza Mahalanobis sono coerenti con il metodo. Resize diretto 224 differisce da resize 256 più center crop 224. Le metriche locali sono vicine a quelle della Tabella III, ma non costituiscono replica esatta. Corrette anche sede e pagine bibliografiche sulla [pagina Springer](https://link.springer.com/chapter/10.1007/978-3-030-68799-1_35).
- **RD4AD:** loss coseno, bottleneck, decoder e principali parametri mantengono il paradigma del paper. Nel [codice ufficiale fissato al commit 6554076](https://github.com/hq-deng/RD4AD/blob/6554076872c65f8784f6ece8cfb39ce77e1aee12/test.py), `evaluation` media PRO per immagine difettosa, mentre qui si usa una curva globale. Anche filtraggio e valutazione sono a risoluzioni diverse (256 contro 1024). Queste sono differenze di protocollo accertate; non è dimostrato il loro contributo numerico allo scarto.
- **FastFlow:** flussi, loss con Jacobiano e mappa negativa esponenziale sono coerenti con il porting Anomalib. Il clone di riferimento è effettivamente `889a67f51cedc943859ae14e52af688a1e1bd1c2`; la formula della mappa coincide con `anomaly_map.py`. L'output negativo non è un errore di orientamento: anomalie hanno valori più alti. Sono osservati zeri in sette mappe, compatibili con il limite numerico della rappresentazione esponenziale; non si attribuisce una causa senza osservare i latenti. Il ricalcolo gestisce correttamente gli ex aequo. Training 200/batch8 differisce dal paper 500/batch32; non è una prova di miglioramento rispetto al paper.

## Correzioni apportate al testo

- Esplicitate formule e differenze cromatiche CCW in capitolo 3, diagnostica in capitolo 4 e portata nelle conclusioni.
- Rimossa la spiegazione non verificata del segno del decoder.
- Sostituita la generica ipotesi sul PRO RD4AD con il riscontro nel codice ufficiale, aggiunto in bibliografia.
- Rimossa l'affermazione di ambienti virtuali separati: il notebook usa `uv run` con extra TensorFlow.
- Versione Keras distinta non dichiarata come registrata: i manifest riportano TensorFlow e il codice usa API Keras.
- Le ricostruzioni CCW sono calcolate in memoria, non presentate come artefatti salvati sistematicamente.
- Hardware non raccolto dichiarato assente, non ricostruito dall'hardware attuale. Rimossi TODO che avrebbero lasciato intendere specifiche già disponibili.

## Controllo delle fonti nei capitoli iniziali

La revisione ha controllato le affermazioni metodologiche nei PDF locali e, dove necessario, nelle pubblicazioni originali e nelle pagine degli editori. Le formule illustrative restano indicate come tali; non vengono attribuite ai quattro percorsi implementativi.

| Affermazioni | Riscontro e correzione |
|---|---|
| Definizione, rarità, eterogeneità, dimensionalità, rumore e interpretabilità | Pang et al., sezioni 2.1–2.2 della [versione degli autori](https://arxiv.org/html/2007.02500v3); non attribuite genericamente al solo ambito industriale. |
| Scarsità di difetti e variabilità di acquisizione industriale | Introduzione di Li et al. e rassegna di Cui; sostituite citazioni a lavori IoT/video che erano meno pertinenti. |
| Normal-only/semi-supervisionato e convenzione one-class | Villa-Pérez et al., sezione 2; Cui e Baitieva per l'uso industriale. Distinta anche la sintesi di anomalie, assente nelle quattro pipeline sperimentate. |
| MVTec AD: 5354 immagini, cinque texture, dieci oggetti, oltre 70 difetti | [Articolo IJCV del 2021](https://link.springer.com/article/10.1007/s11263-020-01400-4), abstract e contributi; la chiave BibTeX storica contiene 2019, ma la voce identifica correttamente l'estensione del 2021. |
| LOCO: 3644 immagini e anomalie strutturali/logiche | [Pagina ufficiale MVTec](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad). |
| AD 2: oltre 8000 immagini, trasparenza, sovrapposizione, variazioni luminose e difetti piccoli | [Articolo IJCV](https://link.springer.com/article/10.1007/s11263-026-02743-0), abstract e sezione 3. Corretto 175 come numero di articolo, non pagine 175–197. |
| Limiti di AUROC e necessità di metriche regionali/a basso FPR | Benchmark di Zheng, fonte MVTec e [AUPIMO, BMVC 2024](https://bmvc2024.org/proceedings/223/); il protocollo locale è verificato nel valutatore, non inferito da questi lavori. |
| SIFT/HOG, densità, OCSVM/SVDD | Yang et al., sezioni introduttive e III. |
| LBP, Gabor, Fourier e wavelet | Cui et al., sezione II; corretta la citazione che rinviava alla rassegna di Tran, pertinente invece a MRF e video. |
| AnoGAN, formula residuo + feature matching | Equazioni 3–5 della [fonte originale](https://arxiv.org/html/1703.05921v1), aggiunta perché la rassegna già citata descriveva il principio ma non forniva il riscontro diretto della formula. |
| CAVGA, DFR, CFA, CutPaste e NSA | Sezioni III di Cui, compreso il Poisson image editing di NSA; DRAEM anche nella fonte originale già citata. |
| Domain gap, adattamento, risorse edge | Li; Roth per coreset; Morshedi mantenuto con ambito IoT dichiarato, senza trasferirne risultati quantitativi all'ispezione di immagini. |
| Continual anomaly detection | PDF Faber: aggiornamenti, forgetting e contaminazione; esplicitato lo stato di preprint non sottoposto a peer review. |
| DCS zero-shot | PDF Wan, abstract e metodo; non presentato come modello sperimentato nel progetto. |

Ulteriori metadati corretti: DOI del tutorial Pang/Cao/Aggarwal `10.1145/3437963.3441659` (prima pagina del PDF locale); anno del preprint Alzarooni **2025**, coerente con la [sottomissione arXiv](https://arxiv.org/abs/2501.11310). Le chiavi bibliografiche preesistenti sono conservate per non introdurre rinomine inutili. Autori, titolo e pagine di Baitieva sono confermati dalla pagina ufficiale CVPR; Zheng include effettivamente Wei Li, verificato nella prima pagina del PDF.

## Verifica finale e limiti residui

- I 64 valori ricalcolati coincidono con gli archivi; i conteggi, le scale e le tabelle del capitolo 4 sono allineati a questi dati e alle tabelle dei paper dichiarate nel testo.
- Compilazione `pdflatex`, `bibtex`, due passaggi `pdflatex`: PDF di 55 pagine, nessun errore, riferimento indefinito o overfull nel log finale. Controllo visivo delle nuove formule, tabella fonte/locale, diagnostica e bibliografia effettuato.
- Le note di lavoro sono aggiornate per non lasciare una spiegazione superata del PRO RD4AD.
- Tutte le 17 verifiche di configurazione e tutte le mappe passano. La versione specifica di Linux è esclusa dai criteri di confronto dell'elaborato su richiesta dell'utente; i risultati del gate automatico e i dettagli di piattaforma sono conservati nei rapporti tecnici originali, senza modificarli.
- Non è necessario un altro test per confermare le metriche archiviate. Rimangono da misurare, come esperimenti futuri e non requisiti soddisfatti da questa revisione, i contributi causali della perdita cromatica, della risposta ai bordi e dei diversi protocolli regionali. Il test su dodici patch non quantifica tali effetti.

Un nuovo training CCW sarebbe necessario per misurare l'effetto di una diversa perdita cromatica, non per confermare le metriche attuali (già ricalcolate). Se proposto, richiederà conferma e dovrà produrre artefatti distinti; nessuna modifica silenziosa dei risultati del confronto.
