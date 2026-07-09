# Contributing

Thanks for contributing! Please follow these steps to propose changes.

1. Fork the repository and create a feature branch from `main`:

```bash
git checkout -b feat/your-feature
```

2. Make changes and keep commits small and focused. Use clear commit messages.

3. Run tests / lint (if any) and ensure the app still runs:

```bash
pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

4. Push your branch to your fork and open a Pull Request against this repository's `main` branch.

5. The repository owners will review and merge after at least one approval.

Guidelines

- Be respectful and descriptive in PR titles and descriptions.
- Do not include secrets or large binary model files in the PR. Upload model files as Releases or use Git LFS.
- If your change affects the app UI or user workflow, include screenshots and a short usage note.

If you are a maintainer and need direct push access, contact the repo owner to be added to the allowed push list.
