# Examples

Three ways to use Civilization, smallest first.

| Example | What it needs | What it shows |
|---|---|---|
| `civilization demo` (CLI, no file) | nothing | the full service path — validation, memory write, retrieval, injection, auditable trace — with a deterministic offline runtime |
| [`remote_client.py`](remote_client.py) | a running service | the dependency-free client: health, capabilities, decision, memory |
| [`embedded_service.py`](embedded_service.py) | a provider, or a local model | hosting the service in your own process and switching backends by configuration |
| [`local_model.py`](local_model.py) | a local model directory | hosting with a local Hugging Face model, no external provider |

Run any of them from the repository root after `pip install -e '.[embedded]'`:

```bash
civilization demo                     # start here; no configuration at all
python examples/remote_client.py      # needs a service (see below)
python examples/embedded_service.py   # needs a provider endpoint and key
python examples/local_model.py        # needs a local model directory
```

To have something to talk to for `remote_client.py`:

```bash
civilization serve --provider-base-url https://provider.example.com/v1 \
  --provider-model your-model        # in one shell
python examples/remote_client.py     # in another
```
