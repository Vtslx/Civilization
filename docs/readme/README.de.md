# Civilization (Deutsch)

Civilization ist eine Forschungslinie, die **Gedächtnis-, Zustands- und Regel-Signale**
(memory, state, rule) aus Prompts herausnimmt und in den Berechnungspfad eines
Sprachmodells verlegt – und das Ergebnis als auditierbaren Dienst mit stabilen SDKs
versioniert.

Die v1-Linie hält ein kleines Sprachmodell **eingefroren** und trainiert darauf einen
Civilization Adapter samt diagnostischer Auslesungen. Gedächtnis, Zustand und Regeln
gelangen als strukturierter Kontext ins Modell, werden in Hidden States projiziert und
lassen sich pfadweise ablatieren und auditieren. Darauf aufbauend folgen eine
Multi-System-Gedächtnisarchitektur, ein persistenter Inferenzdienst, asynchrone Jobs,
Export-Paketierung und anwendungsnahe SDKs.

Dieses Repository ist die **v1-Basis**: die vollständige erste Generation, ohne spätere
Forschungslinien und ohne interne Entwicklungsdokumente.

> Weitere Sprachen: [English](../../README.md) · [简体中文](README.zh-CN.md) ·
> [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) ·
> [Español](README.es.md) · [Français](README.fr.md) · [Русский](README.ru.md)

## Inhalt des Repositories

| Pfad | Beschreibung |
|---|---|
| `src/civilization/research/prototype/` | Erste ausführbare Testbank: Transformer-ähnlicher Kern in NumPy mit Gedächtnis-/Zustands-/Regelvektoren und fusioniertem CivilizationBlock. |
| `src/civilization/research/torch_line/` | PyTorch-Backend-Linie: Logik-Datensätze, Codebooks, Ablationskonfigurationen und die Trainings-/Evaluationsumgebung aller späteren Stufen. |
| `src/civilization/engine/` | Linie mit eingefrorener Basis, Stages 44–160: Hidden-State-Baselines, Civilization Adapter, Training der memory/state/rule-Pfade und die persistente Dienstkette. |
| `src/civilization/` | Python-SDK: abhängigkeitsfreier HTTP-Client, Request-/Prediction-/Job-Modelle, anbieterunabhängige Runtime-Schicht und In-Prozess-Diensthost. |
| `sdk/civilization-transformer/` | TypeScript-SDK: abhängigkeitsfreier Client für Entscheidungen, Gedächtnis, Jobs und Exportpakete. |
| `SDK.md`, `pyproject.toml` | Python-Paketierung für `astreusn-civilization-v1`. |

Über 500 Tests decken die Linie ab – von den ersten Regel-Unit-Tests bis zu den
Dienst-, Gedächtnis- und SDK-Verträgen.

## Architektur

```
                 eingefrorenes Sprachmodell (Gewichte werden nie aktualisiert)
                              │
   Gedächtnis / Zustand / ──►  Civilization Adapter ──► Hidden-State-Auslesungen
        Regel-Signale           (der trainierbare Teil)        │
                              │                               ▼
                              └──────────────►  Entscheidungsauslesung (Optionswahl)
                                                     │
   ┌─────────────────────────────────────────────────┴──────────────────┐
   │  Dienstkette: Inferenz → Zugriffskontrolle → Queue → asynchrone    │
   │  Jobs → Job-Persistenz → Ergebnisspeicher → Batch → Export →       │
   │  Paketierung → Streaming-Auslieferung → Orion-Multi-System-        │
   │  Gedächtnis                                                        │
   └────────────────────────────────────────────────────────────────────┘
                              │
                  Python-SDK / TypeScript-SDK
```

Zwei Entwurfsregeln prägen alles:

1. **Fähigkeiten werden erklärt, nicht angenommen.** Der Dienst meldet über
   `GET /v1/capabilities`, was seine Runtime tatsächlich kann. Ein reiner Textanbieter
   meldet `hidden_states: false` und `adapter_execution: false`; nur die auditierte lokale
   Adapter-Runtime meldet beides als `true`.
2. **Das Entscheidungs-Backend ist Konfiguration, nicht Code.** Jeder
   OpenAI-kompatible Chat-Completions-Endpunkt, jedes lokale Hugging-Face-Causal-Modell
   oder der auditierte lokale Adapter können Produktionsverkehr bedienen.

| Runtime-Typ | Backend | Anbieterunabhängig | Hidden States | Adapter-Ausführung |
|---|---|---|---|---|
| `provider` | beliebiger OpenAI-kompatibler Chat-Completions-Endpunkt | ja | nein | nein |
| `local_transformers` | beliebiges lokales Hugging-Face-Causal-Modell | ja | nein | nein |
| `local_adapter` | der auditierte Civilization Adapter auf festen lokalen Gewichten | ja | ja | ja |

## Versionslinie

| Version | Codename | Stages | Thema | Status |
|---|---|---|---|---|
| `v0.00.01` | Sun | 44–72 | Inferenz mit eingefrorener Basis, strukturierter Adapter, lauffähiger Dienst | implementiert |
| `v0.00.02` | Orion | 73–100 | Multi-System-Gedächtnis (Arbeits-/episodisches/semantisches/prozedurales) | implementiert |
| `v0.00.03` | Trifid | 101–121 | Schnelle episodische Bindung, Hinweis- und Zeitdisambiguierung, kontrolliertes Replay | implementiert |
| `v0.00.04` | Lagoon | 122–132 | Schema-Konsolidierung über Episoden, Provenienz, Konfliktprüfung | implementiert |
| `v0.00.05` | Eagle | 133–138 | Aufgaben-Traces und freigegebenes prozedurales Gedächtnis | implementiert |
| `v0.00.06` | Rosette | 139–150 | Typisierte Kontextpakete, Konfliktbudgets, mehrskaliger Wabengraph | implementiert |
| `v0.00.07` | Helix | 151–160 | Gelernte Abrufpfad-Gewichte mit Wiederherstellung und Kalibrierungsgates | implementiert |
| `v0.00.08` | Crab | — | Erkennung, Schlichtung und Vergessen von Gedächtniskonflikten | **nicht implementiert** |

Release-Labels sind keine Fähigkeitsaussagen. Die v1-Linie reicht bis `v0.00.07`;
`v0.00.08` (Crab) ist ein Plan, und kein Code in diesem Repository implementiert ihn.

Jede implementierte Version hat einen **Baseline-Datensatz** (Umfang, hinzugefügte Fähigkeiten, Vertragsinvarianten, Verifikationsbefehl, Grenzen): [docs/versions](../versions/README.md). Bislang hat nur `v0.00.07` Helix eine aufgezeichnete Messung: [gepaartes A/B des nativen Gedächtnisses](../versions/v0.00.07-helix/experiments/memory-on-off-ab.md) (46 gewonnene, 0 verlorene Paare).

## Schnellstart

### Python-SDK

Der entfernte Client benötigt nur die Standardbibliothek:

```bash
python -m pip install .
```

```python
from civilization import CivilizationClient, CivilizationRequest

client = CivilizationClient("https://civilization.example.com", token_env="CIVILIZATION_API_TOKEN")
prediction = client.predict(
    CivilizationRequest(
        text="Choose the deployment action supported by the verified evidence.",
        answer_options=("approve", "reject"),
        session_id="deployment-42",
        task_name="deployment_decision",
        memory_items=("The deployment signature and health checks are valid.",),
        rule_items=("Reject only when a critical verification is unresolved.",),
        state_values=(0.9, 0.1, 0.8),
    )
)
print(prediction.option_id, prediction.option_text, prediction.memory_trace)
```

Derselbe Client bietet Orion-Gedächtnisoperationen, asynchrone Jobs, Exporte,
Paketauslieferung, Health, Readiness, Metriken und Runtime-Fähigkeiten.

### Dienst im Prozess hosten

```bash
python -m pip install '.[embedded]'
```

```python
from civilization import EmbeddedCivilization, EmbeddedConfig

service = EmbeddedCivilization(
    EmbeddedConfig(
        runtime="provider",
        provider_base_url="https://provider.example.com/v1",
        provider_model="any-chat-model",
        provider_api_key_env="PROVIDER_API_KEY",
    )
)
with service:
    client = service.start()
    print(client.ready(), service.capabilities.to_dict())
```

Wechseln Sie `runtime` auf `local_transformers` (beliebiges lokales Hugging-Face-Modell)
oder `local_adapter` (der auditierte Adapterpfad), ohne die Dienstkette zu ändern. Neue
Backends lassen sich mit `register_runtime(RuntimeKind(...))` registrieren.

### TypeScript-SDK

```bash
cd sdk/civilization-transformer
npm install && npm run build && npm test
```

```ts
import { CivilizationClient } from "@astreusn/civilization-transformer";

const client = new CivilizationClient("https://civilization.example.com", {
  tokenEnv: "CIVILIZATION_API_TOKEN",
});

const capabilities = await client.capabilities();
console.log(capabilities.capabilities?.kind); // "provider" | "local_transformers" | "local_adapter"

const prediction = await client.predict({
  text: "Choose the supported operation.",
  answerOptions: ["approve", "reject"],
  sessionId: "deployment-42",
});
```

## Repository-Struktur

```text
.
├── src/civilization/                     das installierte Paket
│   ├── __init__.py                       öffentliche API
│   ├── client.py                         abhängigkeitsfreier HTTP-Client
│   ├── models.py                         Request-/Prediction-/Job-Modelle
│   ├── runtimes.py                       Runtime-Typen, Fähigkeiten, Registry
│   ├── embedded.py                       In-Prozess-Diensthost
│   ├── cli.py                            `civilization`-Kommandozeile
│   ├── engine/                           Entscheidungs-Engine (engine extras nötig)
│   │   ├── model_paths.py                optionale Auflösung des lokalen Checkpoints
│   │   ├── adapter/                      der trainierbare Civilization Adapter
│   │   ├── backend/                      lokale / Provider- / Adapter-Runtimes
│   │   └── stages/                       versionierte Dienstkette, Stages 44–160
│   └── research/                         frühere Linien, zur Nachvollziehbarkeit erhalten
│       ├── prototype/                    erste NumPy-Testbank
│       └── torch_line/                   PyTorch-Backend-Linie
├── tests/                                Testsuites für engine, torch_line, prototype
├── examples/                             ausführbare Beispiele
├── docs/versions/                        Baseline-Datensatz je Version und Experimente
├── docs/readme/                          dieses README in acht Sprachen
├── sdk/civilization-transformer/         TypeScript-SDK
├── SDK.md                                Python-SDK-Leitfaden
└── pyproject.toml                        Paketierung von astreusn-civilization-v1
```

## Tests

```bash
python -m pip install '.[test]'
pytest
```

Der Großteil der Suite benötigt keine Modellgewichte: Dienstkette, Gedächtnisrichtlinien,
Job- und Exportverträge sowie beide SDKs laufen mit Fake- oder entfernten Runtimes.

Tests, die den **lokalen Qwen3-0.6B-Checkpoint** brauchen, werden bei fehlendem Modell
übersprungen. Der Checkpoint wird nicht mit diesem Repository verteilt; beschaffen Sie ihn
und legen Sie ihn unter `Models/Qwen3-0.6B` ab oder zeigen Sie mit
`CIVILIZATION_MODEL_PATH` darauf:

```bash
CIVILIZATION_MODEL_PATH=/path/to/Qwen3-0.6B pytest
```

Stage-Runner schreiben ihre Ausgaben nach `experiments/*/artifacts/`, das von git
ignoriert wird; nichts im Repository hängt von committeten Experimentausgaben ab.

## Umfang und Aussagegrenzen

Was dieses Repository **ist**:

- Eine versionierte Implementierung der v1-Linie: Inferenz mit eingefrorener Basis,
  strukturierter Adapter, Multi-System-Gedächtnis, Dienstkette und SDKs.
- Ein System, dessen Invarianten getestet sind: fail-closed Eingabevalidierung,
  Zugriffskontrolle, Job-Persistenz und -Wiederherstellung, Exportintegrität
  (SHA-256-verifizierte Downloads) und pfadweise Ablationsaudits.

Was dieses Repository **nicht** beansprucht:

- Es ist kein neues Foundation-Modell. In der v1-Linie bleibt das Basismodell
  eingefroren; trainierbar sind Adapter und Auslesungen.
- Keine Behauptung von lebenslangem Lernen, allgemeinem Langzeitgedächtnis, autonomer
  Fakten-Schlichtung oder Generalisierung auf menschlichem Niveau.
- Keine biologische Äquivalenz. Hirnregionsnamen im Gedächtnisdesign sind
  Ingenieurlabels.
- Keine Behauptung, dass ein reiner Textanbieter den Adapter ausführt oder Hidden States
  offenlegt; der Fähigkeitsendpunkt existiert genau, um diese Aussage zu vermeiden.
- Der auditierte Adapterpfad ist für den festen Checkpoint und die dokumentierten
  Paketartefakte validiert; andere lokale Modelle laufen über den reinen Textpfad.

## Lizenz

Apache License 2.0. Siehe [LICENSE](../../LICENSE) und [NOTICE](../../NOTICE).
