# Kompilera rapporten

Först intallera pygments med `pip install pygments`.  
Därefter behöver flaggan `-shell-escape` läggas till, t.ex. `pdflatex -shell-escape report.tex`.  
För att fungera med VSCode LaTeX Workshop, lägg till i .vscode/settings.json:

```json
{
    "latex-workshop.latex.tools": [
        {
            "name": "pdflatex",
            "command": "pdflatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-shell-escape",
                "%DOC%"
            ]
        }
    ],
    "latex-workshop.latex.recipes": [
        {
            "name": "pdflatex (minted)",
            "tools": [
                "pdflatex"
            ]
        }
    ]
}
```
