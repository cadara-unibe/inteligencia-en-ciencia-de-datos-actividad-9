
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Práctica Integradora de Ciencia de Datos — Sistema de Recomendación
# Script único con bloques lineales por paso.

import os
import argparse
import json
import pickle
from pathlib import Path as _Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template_string

# Configuración y rutas
DATA_PATH = os.environ.get('DATA_PATH', 'data/online_retail_II.xlsx')
OUTPUT_DIR = _Path('outputs'); OUTPUT_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR = _Path('artifacts'); ARTIFACTS_DIR.mkdir(exist_ok=True)

HTML_TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Demo Recomendador</title>
  <style>
    body {font-family: Arial, sans-serif; margin: 2rem;}
    input, button {padding: .5rem;}
    .item {margin: .5rem 0;}
  </style>
</head>
<body>
  <h1>Recomendador de Productos (Online Retail II)</h1>
  <form>
    <label>CustomerID:</label>
    <input type="number" name="customer_id" id="cid" />
    <label>Top K:</label>
    <input type="number" name="k" id="k" value="10" />
    <button type="button" onclick="go()">Recomendar</button>
  </form>
  <div id="res"></div>
  <script>
    async function go(){
      const cid = document.getElementById('cid').value;
      const k = document.getElementById('k').value;
      const r = await fetch(`/recommend?customer_id=${cid}&k=${k}`);
      const data = await r.json();
      const box = document.getElementById('res');
      if(data.error){ box.innerHTML = `<p style='color:red'>${data.error}</p>`; return; }
      box.innerHTML = `<h2>Recomendaciones</h2>` +
        data.recommendations.map(x => `<div class='item'><strong>${x.stockcode}</strong> — ${x.description || ''} (score: ${x.score.toFixed(4)})</div>`).join('');
    }
  </script>
</body>
</html>
"""

# -----------------------------
# Paso 1: Adquisición y limpieza
# -----------------------------
def step_acquire():
    print(f"Cargando datos desde: {DATA_PATH}")
    if not _Path(DATA_PATH).exists():
        print("[ADVERTENCIA] No se encontró el Excel. Descargue 'online_retail_II.xlsx' y colóquelo en ./data. Ver README.md")
        return
    df = pd.read_excel(DATA_PATH, engine='openpyxl')
    print("Forma original:", df.shape)
    df.head(20).to_csv(OUTPUT_DIR / 'before_cleaning_head.csv', index=False)

    # Estándar de columnas
    if 'UnitPrice' in df.columns and 'Price' not in df.columns:
        df = df.rename(columns={'UnitPrice':'Price'})

    df = df.dropna(subset=['CustomerID', 'StockCode'])
    df = df[~df['Invoice'].astype(str).str.startswith(('C','c'))]
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['Price']
    df = df.drop_duplicates()

    print("Forma después de limpieza:", df.shape)
    df.head(20).to_csv(OUTPUT_DIR / 'after_cleaning_head.csv', index=False)
    df.to_parquet(ARTIFACTS_DIR / 'transactions_clean.parquet', index=False)

    items = df[['StockCode','Description']].drop_duplicates().dropna()
    items.to_csv(ARTIFACTS_DIR / 'items.csv', index=False)

    monthly = df.set_index('InvoiceDate').resample('M')['TotalPrice'].sum().reset_index()
    monthly.to_csv(OUTPUT_DIR / 'monthly_revenue.csv', index=False)
    print("Paso 1 completado.")

# -----------------------------
# Paso 2: EDA
# -----------------------------
def step_eda():
    df_path = ARTIFACTS_DIR / 'transactions_clean.parquet'
    if not df_path.exists():
        print("Primero ejecute --step acquire")
        return
    df = pd.read_parquet(df_path)
    sns.set(style='whitegrid')

    top_products = df.groupby(['StockCode','Description'])['TotalPrice'].sum().nlargest(20).reset_index()
    plt.figure(figsize=(12,6))
    sns.barplot(data=top_products, x='TotalPrice', y='Description', palette='viridis')
    plt.title('Top 20 productos por revenue')
    plt.xlabel('Revenue (£)'); plt.ylabel('Producto'); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eda_top_products.png'); plt.close()

    monthly = df.set_index('InvoiceDate').resample('M')['TotalPrice'].sum().reset_index()
    plt.figure(figsize=(12,6))
    sns.lineplot(data=monthly, x='InvoiceDate', y='TotalPrice')
    plt.title('Ventas mensuales'); plt.xlabel('Mes'); plt.ylabel('Revenue (£)'); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eda_sales_over_time.png'); plt.close()

    top_countries = df.groupby('Country')['TotalPrice'].sum().nlargest(10).reset_index()
    plt.figure(figsize=(12,6))
    sns.barplot(data=top_countries, x='TotalPrice', y='Country', palette='magma')
    plt.title('Top 10 países por revenue'); plt.xlabel('Revenue (£)'); plt.ylabel('País'); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eda_countries.png'); plt.close()

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); sns.histplot(df['Price'], bins=50, log_scale=True); plt.title('Distribución de precios (log)')
    plt.subplot(1,2,2); sns.histplot(df['Quantity'], bins=50, log_scale=True); plt.title('Distribución de cantidades (log)')
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'eda_price_quantity_hist.png'); plt.close()

    desc = df[['Quantity','Price','TotalPrice']].describe(); desc.to_csv(OUTPUT_DIR / 'eda_descriptives.csv')
    print("Paso 2 completado.")

# -----------------------------
# Paso 3: NLP (TF-IDF)
# -----------------------------
def step_nlp():
    items_path = ARTIFACTS_DIR / 'items.csv'
    if not items_path.exists():
        print("Primero ejecute --step acquire")
        return
    items = pd.read_csv(items_path)
    items['Description'] = items['Description'].astype(str).str.lower()

    vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
    tfidf = vectorizer.fit_transform(items['Description'])

    pickle.dump(vectorizer, open(ARTIFACTS_DIR / 'tfidf_vectorizer.pkl', 'wb'))
    sparse.save_npz(ARTIFACTS_DIR / 'tfidf_matrix.npz', tfidf)
    items.to_csv(ARTIFACTS_DIR / 'items_vectorized.csv', index=False)

    vocab = np.array(vectorizer.get_feature_names_out())
    mean_tfidf = np.asarray(tfidf.mean(axis=0)).ravel()
    top_idx = mean_tfidf.argsort()[-50:][::-1]
    top_terms = pd.DataFrame({'term': vocab[top_idx], 'mean_tfidf': mean_tfidf[top_idx]})
    top_terms.to_csv(OUTPUT_DIR / 'nlp_top_terms.csv', index=False)
    print("Paso 3 completado.")

# -----------------------------
# Paso 4: Modelado (colaborativo + contenido) y evaluación
# -----------------------------
def _build_interaction_matrix(df):
    user_counts = df.groupby('CustomerID')['StockCode'].nunique()
    df = df[df['CustomerID'].isin(user_counts[user_counts >= 5].index)]
    users = pd.Index(sorted(df['CustomerID'].unique()))
    items = pd.Index(sorted(df['StockCode'].unique()))
    u2i = {u: idx for idx, u in enumerate(users)}
    i2i = {i: idx for idx, i in enumerate(items)}
    rows = df['CustomerID'].map(u2i).values
    cols = df['StockCode'].map(i2i).values
    data = np.ones(len(df), dtype=np.float32)
    mat = sparse.coo_matrix((data, (rows, cols)), shape=(len(users), len(items))).tocsr()
    return mat, users, items

def _train_test_split_last(df):
    df = df.sort_values(['CustomerID','InvoiceDate'])
    last = df.groupby('CustomerID').tail(1)
    train = df.drop(last.index)
    return train, last

def recommend_hybrid_for_user(user_idx, UI_train, item_sim, tfidf_sim, alpha=0.7, k=10):
    u_vec = UI_train[user_idx]
    scores_collab = u_vec.dot(item_sim).A1
    interacted_items = u_vec.indices
    if len(interacted_items) > 0:
        scores_content = tfidf_sim[interacted_items].mean(axis=0).A1
    else:
        scores_content = np.zeros(item_sim.shape[0])
    scores = alpha * scores_collab + (1 - alpha) * scores_content
    scores[interacted_items] = -np.inf
    top_idx = np.argsort(scores)[-k:][::-1]
    return top_idx, scores[top_idx]

def precision_recall_at_k(UI_train, UI_test, item_sim, tfidf_sim, k=10, alpha=0.7):
    hits = 0
    precisions, recalls = [], []
    n_users = UI_train.shape[0]
    for u in range(n_users):
        top_idx, _ = recommend_hybrid_for_user(u, UI_train, item_sim, tfidf_sim, alpha=alpha, k=k)
        test_items = UI_test[u].indices
        if len(test_items) == 0:
            continue
        inter = len(set(top_idx).intersection(set(test_items)))
        hits += int(inter > 0)
        precisions.append(inter / k)
        recalls.append(inter / len(test_items))
    hr = hits / n_users
    prec = float(np.mean(precisions)) if precisions else 0.0
    rec = float(np.mean(recalls)) if recalls else 0.0
    return {'hit_rate@k': hr, 'precision@k': prec, 'recall@k': rec}

def step_model():
    df_path = ARTIFACTS_DIR / 'transactions_clean.parquet'
    tfidf_path = ARTIFACTS_DIR / 'tfidf_matrix.npz'
    items_vec_path = ARTIFACTS_DIR / 'items_vectorized.csv'
    if not (df_path.exists() and tfidf_path.exists() and items_vec_path.exists()):
        print("Primero ejecute --step acquire y --step nlp")
        return
    df = pd.read_parquet(df_path)
    items_df = pd.read_csv(items_vec_path)

    UI, users, items = _build_interaction_matrix(df)
    print("Matriz UI:", UI.shape)

    train_df, test_df = _train_test_split_last(df)
    UI_train, _, _ = _build_interaction_matrix(train_df)
    UI_test, _, _ = _build_interaction_matrix(test_df)

    IU = UI_train.T
    item_sim = cosine_similarity(IU, dense_output=False)

    tfidf = sparse.load_npz(tfidf_path)
    tf_items = pd.Index(items_df['StockCode'].astype(str).values)
    item_codes = items.astype(str)
    present = set(tf_items)
    mask = np.array([code in present for code in item_codes])

    UI_train = UI_train[:, mask]
    UI_test = UI_test[:, mask]
    IU = UI_train.T
    item_sim = cosine_similarity(IU, dense_output=False)
    reorder_idx = [tf_items.get_loc(code) for code in item_codes[mask]]
    tfidf_aligned = tfidf[reorder_idx]
    tfidf_sim = cosine_similarity(tfidf_aligned, dense_output=False)

    metrics = precision_recall_at_k(UI_train, UI_test, item_sim, tfidf_sim, k=10, alpha=0.7)
    print("Métricas (k=10, alpha=0.7):", metrics)
    json.dump(metrics, open(OUTPUT_DIR / 'model_metrics.json', 'w'), indent=2)

    items_masked = item_codes[mask]
    pickle.dump({'items': items_masked}, open(ARTIFACTS_DIR / 'index_maps.pkl', 'wb'))
    sparse.save_npz(ARTIFACTS_DIR / 'item_similarity.npz', item_sim)
    sparse.save_npz(ARTIFACTS_DIR / 'tfidf_similarity.npz', tfidf_sim)
    pickle.dump({'users': np.array(users)}, open(ARTIFACTS_DIR / 'users.pkl', 'wb'))
    print("Paso 4 completado.")

# -----------------------------
# Paso 5: App (Flask)
# -----------------------------
def load_artifacts_for_app():
    idx_maps = pickle.load(open(ARTIFACTS_DIR / 'index_maps.pkl', 'rb'))
    users = pickle.load(open(ARTIFACTS_DIR / 'users.pkl', 'rb'))['users']
    item_sim = sparse.load_npz(ARTIFACTS_DIR / 'item_similarity.npz')
    tfidf_sim = sparse.load_npz(ARTIFACTS_DIR / 'tfidf_similarity.npz')

    df = pd.read_parquet(ARTIFACTS_DIR / 'transactions_clean.parquet')
    user_counts = df.groupby('CustomerID')['StockCode'].nunique()
    df = df[df['CustomerID'].isin(user_counts[user_counts >= 5].index)]
    users_list = pd.Index(sorted(df['CustomerID'].unique()))

    items_masked = pd.Index(idx_maps['items'])
    df = df[df['StockCode'].astype(str).isin(items_masked)]
    u2i = {u: idx for idx, u in enumerate(users_list)}
    i2i = {i: idx for idx, i in enumerate(items_masked)}
    rows = df['CustomerID'].map(u2i)
    cols = df['StockCode'].astype(str).map(i2i)
    mask = rows.notna() & cols.notna()
    UI_train = sparse.coo_matrix((np.ones(mask.sum()), (rows[mask].astype(int), cols[mask].astype(int))),
                                 shape=(len(users_list), len(items_masked))).tocsr()

    items_df = pd.read_csv(ARTIFACTS_DIR / 'items.csv')
    items_df['StockCode'] = items_df['StockCode'].astype(str)
    items_df = items_df.drop_duplicates('StockCode')
    items_df = items_df[items_df['StockCode'].isin(items_masked)]
    items_df = items_df.set_index('StockCode')
    return users_list, items_masked, item_sim, tfidf_sim, UI_train, items_df

def step_app():
    try:
        users_list, items_masked, item_sim, tfidf_sim, UI_train, items_df = load_artifacts_for_app()
    except Exception as e:
        print("Error cargando artefactos:", e)
        print("Ejecute --step model antes del app.")
        return

    app = Flask(__name__)

    def _recommend_for_customer_id(cid, k=10, alpha=0.7):
        try:
            uidx = list(users_list).index(int(cid))
        except ValueError:
            return []
        u_vec = UI_train[uidx]
        scores_collab = u_vec.dot(item_sim).A1
        interacted_items = u_vec.indices
        if len(interacted_items) > 0:
            scores_content = tfidf_sim[interacted_items].mean(axis=0).A1
        else:
            scores_content = np.zeros(item_sim.shape[0])
        scores = alpha * scores_collab + (1 - alpha) * scores_content
        scores[interacted_items] = -np.inf
        top_idx = np.argsort(scores)[-k:][::-1]
        recs = []
        for i in top_idx:
            code = items_masked[i]
            desc = items_df.loc[code]['Description'] if code in items_df.index else ''
            recs.append({'stockcode': str(code), 'description': str(desc), 'score': float(scores[i])})
        return recs

    @app.route('/')
    def home():
        return render_template_string(HTML_TEMPLATE)

    @app.route('/recommend')
    def recommend():
        cid = request.args.get('customer_id'); k = int(request.args.get('k', 10)); alpha = float(request.args.get('alpha', 0.7))
        if cid is None:
            return jsonify({'error': 'Parámetro customer_id requerido'}), 400
        recs = _recommend_for_customer_id(cid, k=k, alpha=alpha)
        if not recs:
            return jsonify({'error': 'CustomerID no encontrado o sin suficientes interacciones.'}), 404
        return jsonify({'customer_id': cid, 'recommendations': recs})

    print("App de Flask iniciada en http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

# Orquestación
def run_all():
    step_acquire()
    step_eda()
    step_nlp()
    step_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Práctica — Recomendador e-commerce')
    parser.add_argument('--step', type=str, default='all', choices=['acquire','eda','nlp','model','app','all'], help='Qué paso ejecutar')
    args = parser.parse_args()
    if args.step == 'acquire': step_acquire()
    elif args.step == 'eda': step_eda()
    elif args.step == 'nlp': step_nlp()
    elif args.step == 'model': step_model()
    elif args.step == 'app': step_app()
    else: run_all()
