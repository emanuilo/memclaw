# Using Memclaw as a Library

The store, index and search work without the agent — only `OPENAI_API_KEY` is needed for embeddings.

```python
import asyncio
from memclaw.config import MemclawConfig
from memclaw.store import MemoryStore
from memclaw.index import MemoryIndex
from memclaw.search import HybridSearch

async def main():
    config = MemclawConfig()
    store = MemoryStore(config)
    index = MemoryIndex(config)
    search = HybridSearch(config, index)

    # Save a memory
    path = store.save("The best pizza in town is at Mario's on 5th Ave")
    await index.index_file(path)

    # Search
    results = await search.search("pizza recommendations")
    for r in results:
        print(f"{r.score:.2f}: {r.content[:80]}")

    index.close()

asyncio.run(main())
```

`MemclawConfig()` reads its settings from the environment and `~/.memclaw/.env`, like the `memclaw` command. To set them in code, pass the group they belong to; anything you leave out is still read from the environment:

```python
from memclaw.config import MemclawConfig, MemorySettings, OpenAISettings

config = MemclawConfig(
    openai=OpenAISettings(api_key="sk-..."),
    memory=MemorySettings(mmr_lambda=0.5),
)
print(config.memory.vector_weight)   # 0.7, the default
```

The groups are `openai`, `memory`, `claude`, `cursor`, `telegram` and `slack`; `agent_backend` and `platform` sit on `MemclawConfig` itself. See `memclaw/config.py`.

See [Memory Architecture](memory-architecture.md) for how chunking, hybrid search, decay and MMR work, and the settings that tune them.
