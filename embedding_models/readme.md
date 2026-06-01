Remember to execute this whenever you add anew implementation

add their path in the .gitmodules

```ini
[submodule "embedding_models/graph2vec"]
    path = embedding_models/graph2vec
    url = https://github.com/benedekrozemberczki/graph2vec.git
```

```bash
git submodule sync
git submodule update --init
```
