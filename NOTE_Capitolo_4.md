# Note di lavoro per il capitolo 4

Analisi limitata a `leather`, integrata dal capitolo 5 di conclusioni. Queste note sono di supporto e non entrano nel PDF.

Per gli approfondimenti successivi su codice, checkpoint e fonti, consultare [VERIFICA_COERENZA_MODELLI.md](VERIFICA_COERENZA_MODELLI.md) e i rapporti separati nella directory `verifiche`.

## Dati e tracciabilità

- Metriche: `../repository_pw/results/{CCW-SSIM,PaDiM,RD4AD,FastFlow}/leather` (file JSON senza estensione).
- Contesto: `../repository_pw/results/confronto_risultati.md`, `comparability_audit.md` e `../repository_pw/COMPARISON_PROTOCOL.md`.
- Mappe: `../repository_pw/results/<modello>/Maps/MVTec_Data/leather/test/`.
- La ricostruzione dei conteggi immagine mantiene una sola soglia globale per modello, quella di massimo F1 sul test. Non si ottimizzano soglie per tipo di difetto.
- Nell'elaborato il sistema operativo è indicato soltanto come Linux; la sua versione specifica non è considerata nel confronto. Non sono stati modificati il gate di audit, i report storici, le configurazioni o i modelli. I dettagli tecnici originari e i controlli sugli artefatti restano conservati nei manifesti e in `figure/capitolo_4/provenienza_figure.json`.

## Figure

Per rigenerarle dalla radice della workspace:

```sh
repository_pw/.venv/bin/python PW_Anomaly_Detection/scripts/figure_capitolo_4.py
```

Il comando verifica gli artefatti, legge i risultati e genera figure; non esegue training. Riscrive le figure e il relativo JSON di provenienza nella directory dedicata al capitolo.

I campioni `000.png` sono scelti in ordine lessicografico per ciascuna tipologia. `good/030.png` è il campione normale con il massimo punteggio CCW-SSIM; `glue/001.png` è il primo falso negativo nella tipologia con più omissioni. Non si dichiarano campioni rappresentativi. Il contorno verde è ground truth; le mappe non sono maschere a soglia. La scala è min–max globale per modello, non per singola immagine e non probabilistica.

## Riscontri nelle fonti già presenti

- PaDiM, Tabella III: `leather`, WR50-Rd550, pixel AUROC 99,2%, PRO 97,8%. Preprocessing originale diverso dal ridimensionamento diretto locale.
- Reverse Distillation, Tabelle 1–2: `leather` a 256 pixel, immagine 100%, pixel 99,4%, PRO 99,1%. Nel codice ufficiale è accertato un PRO per immagine difettosa, diverso dal PRO globale locale; anche la risoluzione di valutazione e filtraggio differisce.
- Reverse Distillation: coincidono i principali parametri dichiarati (200 epoche, batch 16, Adam con learning rate 0,005 e beta 0,5/0,999). Il divario regionale non è una perdita di prestazione misurata a protocollo identico. Non è quantificato quanto ciascuna differenza di procedura contribuisca allo scarto numerico.
- FastFlow, Tabella 2: immagine 100%, pixel 99,5%. Il valore non viene assimilato automaticamente alla configurazione locale WR50. Nessuna affermazione di miglioramento del paper o verifica della velocità di inferenza.
- FastFlow: la fonte descrive 500 epoche e batch 32; il progetto usa 200 epoche e batch 8. Differenza documentata, non prova della causa del vantaggio numerico locale.
- Executive summary di Berasi, sezione 4.5 e Tabella 1: ROC normalizzata fino a FPR 0,3; valore `leather` 0,982. Non confrontabile direttamente con la nostra AUROC pixel completa né con AU-PRO. Il focus del contributo è il confronto tra funzioni di perdita, soprattutto CW-SSIM e CCW-SSIM.
- CCW-SSIM locale: la componente cromatica usa MSE anziché distanza euclidea e pesatura diversa tra training e inferenza. Il test autorizzato su dodici patch documenta una piccola incidenza cromatica sul valore della perdita, non sui gradienti. L'inversione della luminanza non cambia la CW-SSIM e non spiega il divario delle metriche.

## Questioni da discutere nella revisione

1. Quanto spazio dare al compromesso tra omissioni e falsi allarmi, oltre alla graduatoria delle metriche?
2. Se si desidera rafforzare l'analisi caso per caso, calcolare metriche di localizzazione per tipologia su tutte le immagini, mantenendo il protocollo globale e documentando i piccoli campioni.
3. Se si desidera verificare il contributo cromatico originale, progettare un'ablazione CW-SSIM/CCW-SSIM controllata. Il confronto attuale non risponde a questa domanda; l'ablazione non è stata eseguita.
4. Repliche con più seed e misure omogenee di inferenza sono approfondimenti possibili, non risultati disponibili né requisiti già assunti per completare questo caso di studio.

Le fonti esistenti sostengono la discussione attuale. Nuove fonti diventano utili se si sviluppano ulteriormente calibrazione, scelta operativa delle soglie o robustezza; non sono state aggiunte soltanto per ampliare la bibliografia.
