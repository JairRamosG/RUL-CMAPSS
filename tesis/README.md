# Estructura de la Tesis en LaTeX

Esta carpeta contiene el proyecto LaTeX de la tesis. Está alineada con el índice
definido en `_docs/thesis-outline.MD` y `_docs/specs.MD`.

## Estructura de carpetas

```text
tesis/
├── main.tex                          # Archivo principal (compilar este)
├── config/
│   ├── paquetes.tex                  # Paquetes, idioma, tipografía, biblatex
│   └── comandos.tex                  # Comandos personalizados y atajos
├── frontmatter/
│   ├── portada.tex                   # Página de título
│   ├── dedicatoria.tex               # Dedicatoria (opcional)
│   ├── agradecimientos.tex           # Agradecimientos (opcional)
│   ├── resumen.tex                   # Resumen en español
│   └── abstract.tex                  # Abstract en inglés
├── capitulos/
│   ├── 01_introduccion.tex           # Capítulo 1: Introducción General
│   ├── 02_marco_teorico.tex          # Capítulo 2: Marco Teórico y Estado del Arte
│   ├── 03_diseno_experimental.tex    # Capítulo 3: Diseño Experimental
│   ├── 04_resultados.tex             # Capítulo 4: Resultados Experimentales
│   ├── 05_analisis_discusion.tex     # Capítulo 5: Análisis y Discusión
│   └── 06_conclusiones.tex           # Capítulo 6: Conclusiones y Trabajo Futuro
├── figuras/
│   ├── eda/                          # Figuras del análisis exploratorio
│   ├── resultados/                   # Figuras de resultados (RUL real vs predicho)
│   └── arquitectura/                 # Diagramas de pipeline y arquitectura
├── tablas/                           # Tablas (un archivo .tex por tabla)
├── anexos/
│   ├── anexo_a_hiperparametros.tex   # Anexo A: Hiperparámetros
│   ├── anexo_b_codigo.tex            # Anexo B: Código fuente destacado
│   └── anexo_c_resultados.tex        # Anexo C: Resultados adicionales
└── referencias/
    └── bibliografia.bib              # Referencias en formato BibTeX
```

## Cómo compilar

En VSCode con la extensión **LaTeX Workshop**, el build automático se encarga de
todo. La receta recomendada es `latexmk` (que ejecuta las pasadas de `pdflatex`
y `biber` necesarias).

Si compilas manualmente desde la terminal:

```bash
cd tesis
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Convenciones

- **Idioma:** español (paquete `babel`), con abstract en inglés.
- **Bibliografía:** `biblatex` con backend `biber`, estilo `numeric` (IEEE).
  Para APA 7ma, cambiar `style=numeric` por `style=apa` en `config/paquetes.tex`.
- **Figuras:** usar el comando `\figura{ancho}{archivo}{caption}{label}` definido
  en `config/comandos.tex`; las figuras se buscan automáticamente en `figuras/`.
- **Tablas:** un archivo `.tex` por tabla dentro de `tablas/`, incluido con
  `\incluirtabla{archivo}`.
- **Citas:** `\cite{clave}` con las claves definidas en `referencias/bibliografia.bib`.

## Notas

- Las figuras generadas por el EDA y los experimentos viven en `notebooks/figures/`
  y `reports/figures/`. Copiar o enlazar aquí las que se incluyan en la tesis.
- El Anexo B (código) usa el entorno `listings`; ajustar los fragmentos de código
  a incluir cuando se tenga el wrapper de la Memoria Asociativa implementado.
