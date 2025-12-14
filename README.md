¡Claro, CARLOS! Aquí tienes el **README.md** completo dentro de un único bloque para que puedas **copiar y pegar** sin que se rompa el markdown:

````markdown
# Práctica Integradora de Ciencia de Datos — Recomendador de Productos

Este proyecto implementa, de punta a punta, un sistema de recomendación para una empresa de comercio electrónico utilizando el dataset **Online Retail II** del UCI Machine Learning Repository. El flujo incluye adquisición y limpieza de datos, EDA, preprocesamiento de texto (TF‑IDF), modelado (colaborativo + contenido), despliegue con Flask y documentación.

> **Dataset recomendado:** *Online Retail II* (UCI). Contiene transacciones reales de un e‑commerce UK entre 2009 y 2011 y **incluye descripciones de productos** en la columna `Description`.  
> Descarga oficial: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II  \
> Página alternativa (beta): https://archive-beta.ics.uci.edu/dataset/502/online+retail+ii

---

## 1) Preparación del entorno

### Requisitos previos
- Python 3.9+ (recomendado)
- Git (opcional)

### Crear entorno virtual (Windows/macOS/Linux)
```bash
# Crear carpeta del proyecto
mkdir practica-recomendador && cd practica-recomendador

# Crear y activar entorno virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Actualizar pip
python -m pip install --upgrade pip
````

### Instalar dependencias

```bash
# Copia requirements.txt en el directorio y ejecuta:
pip install -r requirements.txt
```

### Descargar y ubicar el dataset

1.  Descarga `online_retail_II.xlsx` desde UCI y colócalo en `data/online_retail_II.xlsx`.
    *   UCI: <https://archive.ics.uci.edu/ml/datasets/Online+Retail+II>
    *   UCI (beta): <https://archive-beta.ics.uci.edu/dataset/502/online+retail+ii>
2.  (Opcional) Si usas otra ruta, define `DATA_PATH`:

```bash
# macOS/Linux
export DATA_PATH=/ruta/a/online_retail_II.xlsx
# Windows (CMD)
set DATA_PATH=C:\ruta\a\online_retail_II.xlsx
```

> **Nota**: El dataset incluye `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice/Price`, `CustomerID`, `Country`. Contiene **valores faltantes** y **cancelaciones** (facturas que inician con ‘C’) que se tratan en el Paso 1. Fuente: UCI Machine Learning Repository (enlaces arriba).

***

## 2) Estructura del proyecto

    practica-recomendador/
    ├── main.py                 # Script único con 5 pasos ejecutables
    ├── requirements.txt        # Dependencias
    ├── README.md               # Esta guía
    ├── data/                   # Coloca aquí online_retail_II.xlsx
    ├── artifacts/              # Artefactos generados (parquet, modelos, similitudes)
    └── outputs/                # Gráficos EDA, muestras antes/después, métricas

***

## 3) Cómo ejecutar cada paso

> Todos los pasos están en `main.py`. Puedes ejecutarlos individualmente con `--step` o correr el pipeline completo con `--step all`.

### Paso 1 — **Adquisición y Limpieza**

```bash
python main.py --step acquire
```

**Qué hace**:

*   Carga el Excel del dataset.
*   Remueve **cancelaciones** (Invoices que empiezan con `C`), **cantidades/precios no positivos**, y **filas sin `CustomerID`**.
*   Convierte `InvoiceDate` a `datetime` y crea `TotalPrice = Quantity * Price`.
*   **Antes/Después**: guarda `outputs/before_cleaning_head.csv` y `outputs/after_cleaning_head.csv`.
*   Exporta base limpia en `artifacts/transactions_clean.parquet`, catálogo de ítems en `artifacts/items.csv`, revenue mensual en `outputs/monthly_revenue.csv`.

### Paso 2 — **EDA**

```bash
python main.py --step eda
```

**Genera** en `outputs/`:

*   `eda_top_products.png`: top 20 productos por revenue.
*   `eda_sales_over_time.png`: revenue mensual.
*   `eda_countries.png`: top 10 países por revenue.
*   `eda_price_quantity_hist.png`: histogramas (log) de `Price` y `Quantity`.
*   `eda_descriptives.csv`: estadísticas descriptivas.

### Paso 3 — **NLP (TF‑IDF)**

```bash
python main.py --step nlp
```

**Procesa**:

*   Limpia texto de `Description` (minúsculas) y vectoriza con `TfidfVectorizer` (stopwords inglés).
*   Guarda `artifacts/tfidf_vectorizer.pkl`, `artifacts/tfidf_matrix.npz`, `artifacts/items_vectorized.csv`.
*   Exporta términos más informativos: `outputs/nlp_top_terms.csv`.

### Paso 4 — **Modelado y Evaluación**

```bash
python main.py --step model
```

**Hace**:

*   Construye matriz **usuario–item** (implícita, conteos) y filtra usuarios con ≥ 5 items.
*   Split por **última interacción** por usuario (train/test).
*   Calcula similitudes:
    *   **Item-based colaborativa** (coseno en matriz *item×usuario*).
    *   **Contenido** (coseno en TF‑IDF de descripciones).
*   Recomienda híbrido con `alpha=0.7` (70% colaborativo, 30% contenido).
*   Métricas: `hit_rate@10`, `precision@10`, `recall@10` → guarda `outputs/model_metrics.json`.
*   Artefactos: `artifacts/item_similarity.npz`, `artifacts/tfidf_similarity.npz`, índices y usuarios (`artifacts/index_maps.pkl`, `artifacts/users.pkl`).

### Paso 5 — **Despliegue (Flask)**

```bash
python main.py --step app
```

**Endpoints**:

*   `/` — formulario HTML para ingresar `CustomerID` y `Top K`.
*   `/recommend?customer_id=...&k=...&alpha=...` — devuelve JSON con recomendaciones.

Abrir en navegador: `http://127.0.0.1:5000`

***

## 4) Explicación del código (resumen por paso)

*   **Acquire**: limpieza rigurosa y exportación de artefactos; muestra comparativa antes/después.
*   **EDA**: usa `seaborn/matplotlib` para entender distribución y tendencias (productos, tiempo, países).
*   **NLP**: representa productos con **TF‑IDF** sobre `Description` para similitud semántica.
*   **Model**: híbrido **colaborativo (item-based)** + **contenido**; evaluación con hold-out de última interacción (métricas `@10`).
*   **App**: Flask consume artefactos, expone HTML y JSON simples.

***

## 5) Ejecución completa (pipeline)

```bash
python main.py --step all
```

Ejecuta **Pasos 1 a 4** de forma secuencial. El **Paso 5 (app)** se corre aparte.

***

## 6) Buenas prácticas

*   El dataset es grande; si tu equipo tiene poca memoria, filtra ítems por frecuencia antes de calcular similitudes.
*   En producción considera métricas adicionales (**MAP**, **NDCG**), pruebas A/B y corrección de sesgos (popularidad, frescura).

***

## 7) Créditos y fuentes

*   **Online Retail II (UCI)** — descripción, columnas y descarga:  
    <https://archive.ics.uci.edu/ml/datasets/Online+Retail+II>  
    <https://archive-beta.ics.uci.edu/dataset/502/online+retail+ii>
