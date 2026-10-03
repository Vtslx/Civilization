# Civilization (Français)

Civilization est une ligne de recherche qui sort les signaux de **mémoire (memory),
d'état (state) et de règles (rule)** des prompts pour les amener dans le chemin de calcul
d'un modèle de langage, puis versionne le résultat sous forme de service auditable avec des
SDK stables.

La ligne v1 garde un petit modèle de langage **gelé** et entraîne par-dessus un
Civilization Adapter et des lectures de diagnostic. Mémoire, état et règles entrent comme
contexte structuré, sont projetés dans les hidden states et peuvent être ablatés et audités
par chemin. Sur ce mécanisme viennent s'ajouter une architecture de mémoire multi-systèmes,
un service d'inférence persistant, des tâches asynchrones, l'empaquetage d'exports et des
SDK applicatifs.

Ce dépôt est la **base v1** : la première génération complète, sans lignes de recherche
ultérieures et sans documents internes de développement.

> Autres langues : [English](../../README.md) · [简体中文](README.zh-CN.md) ·
> [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) ·
> [Español](README.es.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

## Contenu du dépôt

| Chemin | Description |
|---|---|
| `experiments/civilization_transformer/` | Premier banc d'essai exécutable : cœur de type Transformer en NumPy avec vecteurs mémoire/état/règles et un CivilizationBlock fusionné. |
| `experiments/civilization_transformer_torch/` | Ligne backend PyTorch : jeux de données logiques, codebooks, configurations d'ablation et les harnais d'entraînement/évaluation de toutes les étapes suivantes. |
| `experiments/civilization_transformer_qwen3/` | Ligne à base gelée, Stages 44–160 : références de hidden states, Civilization Adapter, entraînement des chemins memory/state/rule et chaîne de service persistante. |
| `civilization_v1/` | SDK Python : client HTTP sans dépendance, modèles requête/prédiction/tâche, couche runtime agnostique du fournisseur et hôte de service en processus. |
| `sdk/civilization-transformer/` | SDK TypeScript : client sans dépendance pour décisions, mémoire, tâches et paquets d'export. |
| `SDK.md`, `pyproject.toml` | Empaquetage Python de `astreusn-civilization-v1`. |

Plus de 500 tests couvrent la ligne, des premiers tests unitaires de règles aux
contrats de service, de mémoire et de SDK.

## Architecture

```
                 modèle de langage gelé (les poids ne sont jamais mis à jour)
                              │
   mémoire / état / règles ──►  Civilization Adapter ──► lectures de hidden state
        (signaux)                (la partie entraînable)       │
                              │                               ▼
                              └──────────────►  lecture de décision (choix d'option)
                                                     │
   ┌─────────────────────────────────────────────────┴──────────────────┐
   │  chaîne de service : inférence → contrôle d'accès → file →         │
   │  tâches asynchrones → persistance → stockage des résultats →       │
   │  lot → export → paquet → diffusion en flux → mémoire Orion         │
   └────────────────────────────────────────────────────────────────────┘
                              │
                  SDK Python / SDK TypeScript
```

Deux règles de conception structurent l'ensemble :

1. **Les capacités sont déclarées, jamais supposées.** Le service indique ce que son
   runtime sait réellement faire via `GET /v1/capabilities`. Un fournisseur texte seul
   déclare `hidden_states: false` et `adapter_execution: false` ; seul le runtime
   d'adapter local audité les déclare à `true`.
2. **Le backend de décision est de la configuration, pas du code.** Tout point d'accès
   Chat Completions compatible OpenAI, tout modèle causal local Hugging Face ou l'adapter
   local audité peuvent servir le trafic de production.

| Type de runtime | Backend | Agnostique du fournisseur | Hidden states | Exécution de l'adapter |
|---|---|---|---|---|
| `provider` | tout point d'accès Chat Completions compatible OpenAI | oui | non | non |
| `local_transformers` | tout modèle causal local Hugging Face | oui | non | non |
| `local_adapter` | l'adapter Civilization audité sur des poids locaux figés | oui | oui | oui |

## Ligne de versions

| Version | Nom de code | Stages | Thème | État |
|---|---|---|---|---|
| `v0.00.01` | Sun | 44–72 | Inférence à base gelée, Adapter structuré, service exécutable | implémenté |
| `v0.00.02` | Orion | 73–100 | Mémoire multi-systèmes (travail / épisodique / sémantique / procédurale) | implémenté |
| `v0.00.03` | Trifid | 101–121 | Liaison épisodique rapide, désambiguïsation par indice et temps, replay contrôlé | implémenté |
| `v0.00.04` | Lagoon | 122–132 | Consolidation de schémas entre épisodes, provenance, revue des conflits | implémenté |
| `v0.00.05` | Eagle | 133–138 | Traces de tâche et mémoire procédurale approuvée | implémenté |
| `v0.00.06` | Rosette | 139–150 | Paquets de contexte typés, budgets de conflit, graphe en nid d'abeille multi-échelle | implémenté |
| `v0.00.07` | Helix | 151–160 | Poids de chemin de récupération appris, avec récupération et portes de calibration | implémenté |
| `v0.00.08` | Crab | — | Détection, arbitrage et oubli des conflits de mémoire | **non implémenté** |

Les étiquettes de publication ne sont pas des affirmations de capacité. La ligne v1
s'arrête à `v0.00.07` ; `v0.00.08` (Crab) est un plan et aucun code de ce dépôt ne
l'implémente.

Chaque version implémentée possède une **fiche de référence** (périmètre, capacités ajoutées, invariants de contrat, commande de vérification, limites) : [docs/versions](../versions/README.md). Une seule version dispose d'une mesure enregistrée, `v0.00.07` Helix : [A/B apparié de la mémoire native](../versions/v0.00.07-helix/experiments/memory-on-off-ab.md) (46 paires gagnées, 0 perdue).

## Démarrage rapide

### SDK Python

Le client distant n'a besoin que de la bibliothèque standard :

```bash
python -m pip install .
```

```python
from civilization_v1 import CivilizationClient, CivilizationRequest

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

Le même client expose les opérations de mémoire Orion, les tâches asynchrones, les
exports, la livraison de paquets, health, readiness, les métriques et les capacités du
runtime.

### Héberger le service en processus

```bash
python -m pip install '.[embedded]'
```

```python
from civilization_v1 import EmbeddedCivilization, EmbeddedConfig

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

Changez `runtime` en `local_transformers` (tout modèle causal local Hugging Face) ou
`local_adapter` (le chemin d'adapter audité) sans toucher à la chaîne de service.
Enregistrez un nouveau backend avec `register_runtime(RuntimeKind(...))`.

### SDK TypeScript

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

## Arborescence

```text
.
├── civilization_v1/                     SDK Python
│   ├── client.py                        client HTTP sans dépendance
│   ├── models.py                        modèles requête / prédiction / tâche
│   ├── runtimes.py                      types de runtime, capacités, registre
│   └── embedded.py                      hôte de service en processus
├── experiments/
│   ├── civilization_transformer/        premier banc d'essai NumPy
│   ├── civilization_transformer_torch/  ligne backend PyTorch
│   └── civilization_transformer_qwen3/  ligne à base gelée, Stages 44–160
│       ├── adapter/                     Civilization Adapter
│       ├── backend/                     backend Qwen3 et runtimes fournisseur
│       ├── analysis/                    exécuteurs d'étape et chaîne de service
│       ├── tests/                       contrats de chaque étape
│       └── model_paths.py               résolution du checkpoint (voir Tests)
├── sdk/civilization-transformer/        SDK TypeScript
├── SDK.md                               guide du SDK Python
└── pyproject.toml                       empaquetage de astreusn-civilization-v1
```

## Tests

```bash
python -m pip install '.[test]'
pytest experiments -q
```

La majeure partie de la suite ne nécessite aucun poids de modèle : la chaîne de service,
les politiques de mémoire, les contrats de tâches et d'exports et les deux SDK
fonctionnent avec des runtimes factices ou distants.

Les tests qui nécessitent le **checkpoint local Qwen3-0.6B** sont ignorés lorsqu'il est
absent. Le checkpoint n'est pas distribué avec ce dépôt ; obtenez-le puis placez-le dans
`Models/Qwen3-0.6B` ou pointez `CIVILIZATION_MODEL_PATH` vers lui :

```bash
CIVILIZATION_MODEL_PATH=/path/to/Qwen3-0.6B pytest experiments -q
```

Les exécuteurs d'étape écrivent leurs sorties dans `experiments/*/artifacts/`, ignoré par
git ; rien dans le dépôt ne dépend de résultats d'expériences déjà commités.

## Portée et limites d'affirmation

Ce que ce dépôt **est** :

- Une implémentation versionnée de la ligne v1 : inférence à base gelée, adapter
  structuré, mémoire multi-systèmes, chaîne de service et SDK.
- Un système dont les invariants sont testés : validation d'entrée fail-closed, contrôle
  d'accès, persistance et récupération des tâches, intégrité des exports (téléchargements
  vérifiés par SHA-256) et audit d'ablation par chemin.

Ce que ce dépôt **n'affirme pas** :

- Ce n'est pas un nouveau modèle de fondation. Dans la ligne v1, le modèle de base reste
  gelé ; la partie entraînable est l'Adapter et ses lectures.
- Aucune affirmation d'apprentissage continu, de mémoire à long terme générale,
  d'arbitrage autonome des faits ou de généralisation de niveau humain.
- Aucune équivalence biologique. Les noms de régions cérébrales sont des étiquettes
  d'ingénierie.
- Aucune affirmation qu'un fournisseur texte seul exécute l'adapter ou expose des hidden
  states ; le point d'accès de capacités existe précisément pour éviter cette confusion.
- Le chemin d'adapter audité est validé pour le checkpoint figé et les artefacts de paquet
  enregistrés ; les autres modèles locaux passent par le chemin texte seul.

## Licence

Apache License 2.0. Voir [LICENSE](../../LICENSE) et [NOTICE](../../NOTICE).
