# Neural Project — Riconoscimento comandi vocali

Breve README per eseguire e valutare i modelli su SpeechCommands.

**Contenuto**
- Panoramica
- Dataset
- Addestramento
- Valutazione e ispezione degli errori
- Inferenza su file singoli

Panoramica
---------
Progetto di riconoscimento vocale basato su PyTorch / torchaudio. I file principali:
- dataset.py — pipeline e filtri per le 10 classi usate
- train.py — script di addestramento
- evaluate.py — valutazione e generazione CSV di errori
- inspect_wrong.py — ispeziona/copia i WAV misclassificati
- file_inference.py — inference su singoli WAV


Dataset
-------
Il dataset usato è Google SpeechCommands (v0.02) sotto `dataset_audio/SpeechCommands/speech_commands_v0.02`.

Questo progetto usa una sottoselezione di 10 etichette per training/eval:
`['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']`

Addestramento
-------------
Esempio per avviare l'addestramento (configura parametri in `train.py`):

```bash
train.py --model light
```

Valutazione
-----------
Esegue la valutazione su tutte le 10 classi e scrive i risultati in `eval_out`.
Genera anche `eval_out/wrong_<model>.csv` con le colonne `file,path,true,pred,pred_confidence`.

```bash
evaluate.py light
```

Ispezione degli errori
----------------------
Lo script `inspect_wrong.py` legge il CSV generato e copia i WAV misclassificati in `eval_out/wrong_samples/` per ispezione manuale:

```bash
inspect_wrong.py --model light --copy
```

Inferenza su file singoli
------------------------
Per inferire singoli file WAV o una cartella di WAV:

```bash
file_inference.py path/to/file_or_dir --model light
```
