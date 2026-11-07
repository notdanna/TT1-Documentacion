"""
clip_filtro_tipo_imagen.py

Filtra automaticamente imagenes que NO son corteza/tronco en primer
plano (follaje, arbol completo, u otro) usando embeddings de CLIP,
antes de que entren al dataset de entrenamiento del clasificador
sano/infestado.

Mejoras respecto a un CLIP "zero-shot" simple (lo que ya probaste y no
fue suficiente):

1. Ensamble de prompts por categoria en vez de un solo prompt. CLIP es
   sensible a la redaccion exacta del texto; promediar varias
   formulaciones por categoria reduce ese ruido (tecnica del propio
   paper de CLIP, "prompt ensembling").
2. Modelo mas grande (ViT-L/14 en vez de ViT-B/32): mejora notable en
   clasificacion fina, a costa de mas computo por imagen -- aceptable
   porque esto se corre una sola vez, no en produccion.
3. Clasificacion multi-categoria explicita (tronco / follaje /
   arbol_completo / otro), no binaria -- separa los tres problemas que
   mencionas, no solo "es tronco o no".
4. Zona de incertidumbre: si el margen entre la mejor y la segunda
   mejor categoria es chico, la imagen se marca "revisar" en vez de
   forzar una decision. Mejor abstenerse que decidir mal con seguridad
   artificial.
5. LA MEJORA MAS GRANDE (opcional, requiere una muestra chica
   etiquetada a mano): si le das --seed_labels con 200-300 imagenes
   corregidas por categoria, el script entrena una regresion logistica
   SOBRE los embeddings de CLIP ya calculados. Normalmente supera por
   mucho al zero-shot y entrena en segundos.

FLUJO RECOMENDADO (minimiza el trabajo manual):

    Paso 1 -- zero-shot, sin etiquetar nada:
        python clip_filtro_tipo_imagen.py --data_dir ./dataset_sin_filtrar \
            --out tipos_zero_shot.csv

    Paso 2 -- toma una muestra guiada por el paso 1 (p. ej. 50-100 por
    categoria predicha, usando pandas.groupby(...).sample(...)),
    corrigela a mano en una hoja de calculo (mucho mas rapido que
    etiquetar desde cero porque ya vienen pre-ordenadas), guardala como
    CSV con columnas "filepath,tipo" y vuelve a correr:

        python clip_filtro_tipo_imagen.py --data_dir ./dataset_sin_filtrar \
            --out tipos_linear_probe.csv \
            --seed_labels muestra_corregida.csv

Instalar dependencias:
    pip install transformers torch pillow scikit-learn pandas numpy
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.linear_model import LogisticRegression

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CATEGORIAS = ["tronco", "follaje", "arbol_completo", "otro"]

# Varios prompts por categoria -- se promedian sus embeddings de texto.
# En ingles a proposito: CLIP fue entrenado mayoritariamente con
# descripciones en ingles y su desempeno zero-shot suele ser mejor con
# prompts en ese idioma, incluso para conceptos que no son en si
# "de habla inglesa".
PROMPTS_POR_CATEGORIA = {
    "tronco": [
        "a close-up photo of tree bark",
        "a macro photo of tree trunk texture",
        "a detailed photo of the bark surface of a tree",
        "a zoomed-in photo of tree bark with visible texture",
    ],
    "follaje": [
        "a photo of green tree leaves and foliage",
        "a close-up photo of a tree canopy",
        "a photo of leaves on tree branches",
        "a photo of tree foliage from below",
    ],
    "arbol_completo": [
        "a photo of a whole tree from a distance",
        "a full view photo of an entire tree from base to top",
        "a landscape photo showing a complete standing tree",
        "a wide shot of a single tree in a forest",
    ],
    "otro": [
        "a photo of a forest landscape without a clear subject",
        "a photo of a person or people outdoors",
        "a generic outdoor photo unrelated to tree bark or foliage",
        "a photo of the ground, sky, or unrelated object",
    ],
}


def elegir_dispositivo() -> str:
    """
    Orden de preferencia: GPU Nvidia (CUDA) > GPU de Apple Silicon
    (Metal, via el backend MPS de PyTorch) > CPU.

    El script original solo checaba CUDA, por eso en Mac siempre caia
    a CPU aunque hubiera GPU disponible. MPS requiere macOS 12.3+ y una
    Mac con Apple Silicon (M1/M2/M3/M4); en Mac Intel no hay MPS y se
    usa CPU de todas formas.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extraer_embedding(salida):
    """
    A partir de transformers 5.x, CLIPModel.get_image_features() y
    get_text_features() dejaron de regresar un tensor plano: ahora
    regresan un objeto BaseModelOutputWithPooling, y el embedding YA
    PROYECTADO queda en el atributo .pooler_output (reutilizan ese
    nombre en vez de uno nuevo). En transformers <5.0 seguian
    regresando el tensor directo. Esta funcion soporta ambos casos sin
    que tengas que fijarte en que version tienes instalada.
    """
    return salida.pooler_output if hasattr(salida, "pooler_output") else salida


def cargar_modelo(nombre_modelo: str, dispositivo: str):
    print(f"Cargando {nombre_modelo} en {dispositivo}...")
    modelo = CLIPModel.from_pretrained(nombre_modelo).to(dispositivo).eval()
    procesador = CLIPProcessor.from_pretrained(nombre_modelo)
    return modelo, procesador


def listar_imagenes(data_dir: Path) -> list:
    return [
        p for p in sorted(data_dir.rglob("*"))
        if p.suffix.lower() in VALID_EXTENSIONS
    ]


@torch.no_grad()
def calcular_embeddings_texto(modelo, procesador, dispositivo) -> dict:
    """Un embedding promedio (normalizado) por categoria, resultado del ensamble de prompts."""
    embeddings = {}
    for categoria, prompts in PROMPTS_POR_CATEGORIA.items():
        entradas = procesador(text=prompts, return_tensors="pt", padding=True).to(dispositivo)
        emb = _extraer_embedding(modelo.get_text_features(**entradas))
        emb = emb / emb.norm(dim=-1, keepdim=True)
        promedio = emb.mean(dim=0)
        embeddings[categoria] = promedio / promedio.norm()
    return embeddings


@torch.no_grad()
def calcular_embeddings_imagenes(
    rutas: list, modelo, procesador, dispositivo, tam_lote: int = 32
) -> np.ndarray:
    """
    Procesa en lotes (no imagen por imagen) porque con miles de
    archivos la diferencia de velocidad es enorme, sobre todo en CPU.
    Las imagenes que no se puedan abrir se devuelven como vector de
    ceros y se marcan aparte -- no truenan todo el proceso.
    """
    embeddings = []
    validas = []
    for i in range(0, len(rutas), tam_lote):
        lote_rutas = rutas[i : i + tam_lote]
        imgs, ok = [], []
        for r in lote_rutas:
            try:
                imgs.append(Image.open(r).convert("RGB"))
                ok.append(True)
            except Exception:
                imgs.append(Image.new("RGB", (224, 224)))
                ok.append(False)

        entradas = procesador(images=imgs, return_tensors="pt").to(dispositivo)
        emb = _extraer_embedding(modelo.get_image_features(**entradas))
        emb = emb / emb.norm(dim=-1, keepdim=True)

        embeddings.append(emb.cpu().numpy())
        validas.extend(ok)
        if (i // tam_lote) % 10 == 0:
            print(f"  procesadas {min(i + tam_lote, len(rutas))}/{len(rutas)}")

    return np.concatenate(embeddings, axis=0), np.array(validas)


def clasificar_zero_shot(embeddings_img: np.ndarray, embeddings_texto: dict, temperatura: float):
    categorias = list(embeddings_texto.keys())
    matriz_texto = np.stack([embeddings_texto[c].cpu().numpy() for c in categorias])  # (C, D)
    sims = embeddings_img @ matriz_texto.T  # (N, C)
    logits = sims * temperatura
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs, categorias


def entrenar_linear_probe(
    df_seed: pd.DataFrame,
    filepaths: list,
    embeddings_img: np.ndarray,
    categoria_objetivo: Optional[str] = None,
):
    """
    Entrena una regresion logistica SOBRE los embeddings de CLIP ya
    calculados para las imagenes de la muestra etiquetada a mano. No
    vuelve a pasar las imagenes por CLIP -- por eso es tan rapido.

    categoria_objetivo:
        Si se especifica (p.ej. "tronco"), convierte el problema a
        BINARIO: esa categoria contra todo lo demas. Con pocas
        imagenes etiquetadas (algunas decenas por clase), calibrar UNA
        frontera de decision binaria es mas facil y da margenes de
        confianza mas utiles que calibrar 4 fronteras simultaneas --
        y suele ser lo que en realidad te importa: aislar una
        categoria especifica, no clasificar perfectamente las otras
        tres entre si.
    """
    indice = {fp: i for i, fp in enumerate(filepaths)}
    idx_seed = [indice[fp] for fp in df_seed["filepath"] if fp in indice]
    faltantes = len(df_seed) - len(idx_seed)
    if faltantes:
        print(f"Aviso: {faltantes} rutas de seed_labels no se encontraron en data_dir, se ignoran.")

    X = embeddings_img[idx_seed]
    y = df_seed["tipo"].values[: len(idx_seed)]

    if categoria_objetivo:
        y = np.where(y == categoria_objetivo, categoria_objetivo, f"no_{categoria_objetivo}")
        print(f"Modo binario activado: '{categoria_objetivo}' vs. el resto.")

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X, y)

    print("Precision del linear probe sobre su propia muestra de entrenamiento:")
    print(f"  {clf.score(X, y):.3f}")
    return clf


def clasificar_linear_probe(clf, embeddings_img: np.ndarray):
    probs = clf.predict_proba(embeddings_img)
    categorias = list(clf.classes_)
    return probs, categorias


def construir_resultado(rutas, validas, probs, categorias, umbral_margen: float) -> pd.DataFrame:
    idx_ordenado = np.argsort(-probs, axis=1)
    top1 = idx_ordenado[:, 0]
    top2 = idx_ordenado[:, 1] if probs.shape[1] > 1 else idx_ordenado[:, 0]

    filas = []
    for i, ruta in enumerate(rutas):
        fila = {"filepath": str(ruta), "imagen_valida": bool(validas[i])}
        if validas[i]:
            confianza = probs[i, top1[i]]
            margen = confianza - probs[i, top2[i]]
            fila["tipo_predicho"] = categorias[top1[i]]
            fila["confianza"] = float(confianza)
            fila["margen"] = float(margen)
            fila["revisar_manual"] = bool(margen < umbral_margen)
            for c_idx, c_nombre in enumerate(categorias):
                fila[f"prob_{c_nombre}"] = float(probs[i, c_idx])
        else:
            fila["tipo_predicho"] = None
            fila["confianza"] = None
            fila["margen"] = None
            fila["revisar_manual"] = True
        filas.append(fila)
    return pd.DataFrame(filas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("tipos_imagen.csv"))
    parser.add_argument("--modelo", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument(
        "--seed_labels",
        type=Path,
        default=None,
        help="CSV opcional con columnas filepath,tipo para activar el modo linear probe.",
    )
    parser.add_argument(
        "--umbral_margen",
        type=float,
        default=0.20,
        help="Si la diferencia entre la mejor y segunda mejor categoria es menor a esto, se marca revisar_manual=True.",
    )
    parser.add_argument(
        "--categoria_objetivo",
        type=str,
        default=None,
        help=(
            "Si se especifica (p.ej. 'tronco'), convierte la clasificacion a "
            "binaria: esa categoria vs. todo lo demas. Recomendado cuando "
            "solo te importa aislar UNA categoria y el modo de 4 clases da "
            "demasiadas imagenes marcadas para revision manual."
        ),
    )
    parser.add_argument("--tam_lote", type=int, default=32)
    parser.add_argument("--cache_embeddings", type=Path, default=Path("embeddings_clip.npy"))
    args = parser.parse_args()

    dispositivo = elegir_dispositivo()
    modelo, procesador = cargar_modelo(args.modelo, dispositivo)

    rutas = listar_imagenes(args.data_dir)
    print(f"Encontradas {len(rutas)} imagenes en {args.data_dir}")

    if args.cache_embeddings.exists():
        print(f"Reutilizando embeddings en cache: {args.cache_embeddings}")
        embeddings_img = np.load(args.cache_embeddings)
        validas = np.ones(len(rutas), dtype=bool)  # asumido; recalcula si cambio el dataset
    else:
        embeddings_img, validas = calcular_embeddings_imagenes(
            rutas, modelo, procesador, dispositivo, args.tam_lote
        )
        np.save(args.cache_embeddings, embeddings_img)
        print(f"Embeddings guardados en cache: {args.cache_embeddings}")

    if args.seed_labels and args.seed_labels.exists():
        print(f"\nModo LINEAR PROBE usando {args.seed_labels}")
        df_seed = pd.read_csv(args.seed_labels)
        rutas_str = [str(r) for r in rutas]
        clf = entrenar_linear_probe(df_seed, rutas_str, embeddings_img, args.categoria_objetivo)
        probs, categorias = clasificar_linear_probe(clf, embeddings_img)
    else:
        print("\nModo ZERO-SHOT (ensamble de prompts, sin --seed_labels)")
        embeddings_texto = calcular_embeddings_texto(modelo, procesador, dispositivo)
        temperatura = modelo.logit_scale.exp().item()  # escala aprendida por el propio CLIP
        probs, categorias = clasificar_zero_shot(embeddings_img, embeddings_texto, temperatura)

    df_resultado = construir_resultado(rutas, validas, probs, categorias, args.umbral_margen)
    df_resultado.to_csv(args.out, index=False)

    print(f"\nResultado guardado en {args.out}")
    print(df_resultado["tipo_predicho"].value_counts(dropna=False).to_string())
    print(f"\nMarcadas para revision manual (margen < {args.umbral_margen}): "
          f"{df_resultado['revisar_manual'].sum()} de {len(df_resultado)}")


if __name__ == "__main__":
    main()
