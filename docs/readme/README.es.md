# Civilization (Español)

Civilization es una línea de investigación que saca las señales de
**memoria (memory), estado (state) y reglas (rule)** de los prompts y las lleva al
propio camino de cómputo de un modelo de lenguaje; después versiona el resultado como un
servicio auditable con SDKs estables.

La línea v1 mantiene **congelado** un modelo de lenguaje pequeño y entrena sobre él un
Civilization Adapter y lecturas de diagnóstico. La memoria, el estado y las reglas entran
como contexto estructurado, se proyectan en los hidden states y se pueden ablacionar y
auditar por ruta. Sobre ese mecanismo se añaden una arquitectura de memoria multisistema,
un servicio de inferencia persistente, trabajos asíncronos, empaquetado de exportaciones y
SDKs para aplicaciones.

Este repositorio es la **base v1**: la primera generación completa, sin líneas de
investigación posteriores y sin documentos internos de desarrollo.

> Otros idiomas: [English](../../README.md) · [简体中文](README.zh-CN.md) ·
> [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) ·
> [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

## Contenido del repositorio

| Ruta | Descripción |
|---|---|
| `experiments/civilization_transformer/` | Primer banco de pruebas ejecutable: núcleo tipo Transformer en NumPy con vectores de memoria/estado/reglas y un CivilizationBlock fusionado. |
| `experiments/civilization_transformer_torch/` | Línea de backend PyTorch: conjuntos de datos lógicos, codebooks, configuraciones de ablación y los arneses de entrenamiento/evaluación de todas las etapas posteriores. |
| `experiments/civilization_transformer_qwen3/` | Línea de base congelada, Stages 44–160: líneas base de hidden states, Civilization Adapter, entrenamiento de rutas memory/state/rule y la cadena de servicio persistente. |
| `civilization_v1/` | SDK de Python: cliente HTTP sin dependencias, modelos de petición/predicción/trabajo, capa de runtime agnóstica del proveedor y host de servicio en proceso. |
| `sdk/civilization-transformer/` | SDK de TypeScript: cliente sin dependencias para decisiones, memoria, trabajos y paquetes de exportación. |
| `SDK.md`, `pyproject.toml` | Empaquetado Python de `astreusn-civilization-v1`. |

Más de 500 pruebas cubren la línea, desde las primeras pruebas unitarias de reglas
hasta los contratos de servicio, memoria y SDK.

## Arquitectura

```
                 modelo de lenguaje congelado (los pesos nunca se actualizan)
                              │
   memoria / estado / reglas ──►  Civilization Adapter ──► lecturas de hidden state
        (señales)                 (la parte entrenable)          │
                              │                                 ▼
                              └──────────────►  lectura de decisión (elección de opción)
                                                     │
   ┌─────────────────────────────────────────────────┴──────────────────┐
   │  cadena de servicio: inferencia → control de acceso → cola →       │
   │  trabajos asíncronos → persistencia → almacén de resultados →      │
   │  lote → exportación → paquete → entrega en streaming →             │
   │  memoria multisistema Orion                                        │
   └────────────────────────────────────────────────────────────────────┘
                              │
                  SDK de Python / SDK de TypeScript
```

Dos reglas de diseño lo atraviesan todo:

1. **Las capacidades se declaran, no se suponen.** El servicio informa lo que su runtime
   puede hacer realmente en `GET /v1/capabilities`. Un proveedor solo-texto declara
   `hidden_states: false` y `adapter_execution: false`; solo el runtime de adapter local
   auditado las declara como `true`.
2. **El backend de decisión es configuración, no código.** Cualquier endpoint Chat
   Completions compatible con OpenAI, cualquier modelo causal local de Hugging Face o el
   adapter local auditado pueden servir tráfico de producción.

| Tipo de runtime | Backend | Agnóstico del proveedor | Hidden states | Ejecución de adapter |
|---|---|---|---|---|
| `provider` | cualquier endpoint Chat Completions compatible con OpenAI | sí | no | no |
| `local_transformers` | cualquier modelo causal local de Hugging Face | sí | no | no |
| `local_adapter` | el Civilization Adapter auditado sobre pesos locales fijados | sí | sí | sí |

## Línea de versiones

| Versión | Nombre | Stages | Tema | Estado |
|---|---|---|---|---|
| `v0.00.01` | Sun | 44–72 | Inferencia con base congelada, Adapter estructurado, servicio ejecutable | implementado |
| `v0.00.02` | Orion | 73–100 | Memoria multisistema (trabajo / episódica / semántica / procedimental) | implementado |
| `v0.00.03` | Trifid | 101–121 | Enlace episódico rápido, desambiguación por señal y tiempo, replay controlado | implementado |
| `v0.00.04` | Lagoon | 122–132 | Consolidación de esquemas entre episodios, procedencia, revisión de conflictos | implementado |
| `v0.00.05` | Eagle | 133–138 | Trazas de tarea y memoria procedimental aprobada | implementado |
| `v0.00.06` | Rosette | 139–150 | Paquetes de contexto tipados, presupuestos de conflicto, grafo de panal multiescala | implementado |
| `v0.00.07` | Helix | 151–160 | Pesos de ruta de recuperación aprendidos, con recuperación y puertas de calibración | implementado |
| `v0.00.08` | Crab | — | Detección, arbitraje y olvido de conflictos de memoria | **no implementado** |

Las etiquetas de publicación no son afirmaciones de capacidad. La línea v1 llega hasta
`v0.00.07`; `v0.00.08` (Crab) es un plan y ningún código de este repositorio lo implementa.

Cada versión implementada tiene un **registro de base** (alcance, capacidades añadidas, invariantes de contrato, comando de verificación y límites): [docs/versions](../versions/README.md). Solo una versión tiene una medición registrada, `v0.00.07` Helix: [A/B pareado de memoria nativa](../versions/v0.00.07-helix/experiments/memory-on-off-ab.md) (46 pares ganados y 0 perdidos).

## Inicio rápido

### SDK de Python

El cliente remoto solo necesita la biblioteca estándar:

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

El mismo cliente expone operaciones de memoria Orion, trabajos asíncronos, exportaciones,
entrega de paquetes, health, readiness, métricas y capacidades del runtime.

### Alojar el servicio en proceso

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

Cambie `runtime` a `local_transformers` (cualquier modelo causal local de Hugging Face) o
`local_adapter` (la ruta de adapter auditada) sin tocar la cadena de servicio. Registre un
backend nuevo con `register_runtime(RuntimeKind(...))`.

### SDK de TypeScript

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

## Estructura del repositorio

```text
.
├── civilization_v1/                     SDK de Python
│   ├── client.py                        cliente HTTP sin dependencias
│   ├── models.py                        modelos de petición / predicción / trabajo
│   ├── runtimes.py                      tipos de runtime, capacidades, registro
│   └── embedded.py                      host de servicio en proceso
├── experiments/
│   ├── civilization_transformer/        primer banco de pruebas NumPy
│   ├── civilization_transformer_torch/  línea de backend PyTorch
│   └── civilization_transformer_qwen3/  línea de base congelada, Stages 44–160
│       ├── adapter/                     Civilization Adapter
│       ├── backend/                     backend Qwen3 y runtimes de proveedor
│       ├── analysis/                    ejecutores de etapa y cadena de servicio
│       ├── tests/                       contratos de cada etapa
│       └── model_paths.py               resolución del checkpoint (ver Pruebas)
├── sdk/civilization-transformer/        SDK de TypeScript
├── SDK.md                               guía del SDK de Python
└── pyproject.toml                       empaquetado de astreusn-civilization-v1
```

## Pruebas

```bash
python -m pip install '.[test]'
pytest experiments -q
```

La mayoría de la suite no necesita pesos de modelo: la cadena de servicio, las políticas de
memoria, los contratos de trabajos y exportaciones y ambos SDK funcionan con runtimes falsos
o remotos.

Las pruebas que necesitan el **checkpoint local Qwen3-0.6B** se omiten cuando no está
presente. El checkpoint no se distribuye con este repositorio; obténgalo y colóquelo en
`Models/Qwen3-0.6B` o apunte `CIVILIZATION_MODEL_PATH` hacia él:

```bash
CIVILIZATION_MODEL_PATH=/path/to/Qwen3-0.6B pytest experiments -q
```

Los ejecutores de etapa escriben sus salidas en `experiments/*/artifacts/`, que está
ignorado por git; nada en el repositorio depende de resultados de experimentos ya cometidos.

## Alcance y límites de afirmación

Lo que este repositorio **es**:

- Una implementación versionada de la línea v1: inferencia con base congelada, adapter
  estructurado, memoria multisistema, cadena de servicio y SDKs.
- Un sistema cuyos invariantes están probados: validación de entrada fail-closed, control
  de acceso, persistencia y recuperación de trabajos, integridad de exportaciones
  (descargas verificadas con SHA-256) y auditoría de ablación por ruta.

Lo que este repositorio **no** afirma:

- No es un nuevo modelo fundacional. En la línea v1 el modelo base permanece congelado; la
  parte entrenable es el Adapter y sus lecturas.
- No afirma aprendizaje de por vida, memoria a largo plazo general, arbitraje autónomo de
  hechos ni generalización a nivel humano.
- No afirma equivalencia biológica. Los nombres de regiones cerebrales son etiquetas de
  ingeniería.
- No afirma que un proveedor solo-texto ejecute el adapter o exponga hidden states; el
  endpoint de capacidades existe precisamente para evitar esa afirmación.
- La ruta de adapter auditada está validada para el checkpoint fijado y los artefactos de
  paquete registrados; otros modelos locales usan la ruta solo-texto.

## Licencia

Apache License 2.0. Véase [LICENSE](../../LICENSE) y [NOTICE](../../NOTICE).
