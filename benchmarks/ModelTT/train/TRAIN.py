"""
entrenamiento_efficientnetb3.py

Entrena un clasificador binario de corteza (sano / infestado) con
transfer learning en dos fases sobre EfficientNet-B3.

Tres decisiones de diseño clave, motivadas por corridas anteriores que
mostraban un sesgo sistemático hacia predecir "infestado":

1. EarlyStopping/ModelCheckpoint ya NO monitorean solo val_recall.
   val_recall en un problema binario con sigmoid mide únicamente el
   recall de la clase 1 (infestado) e ignora por completo cuántos
   sanos se están clasificando mal. Un modelo que dice "infestado"
   casi siempre tiene val_recall altísimo y aun así es un mal modelo.
   Ahora se monitorea val_fbeta_infestado, calculado por un callback
   propio (MetricasValidacionCompletas) que SÍ toma en cuenta la clase
   sano.

2. Los pesos de clase ya no aplican el balanceo completo de sklearn
   por default -- se atenúan con --suavizado_pesos, porque el
   desbalance real (3,452 sano vs 2,634 infestado, ~1.3:1) es leve y
   el balanceo completo ya empuja de más hacia infestado.

3. El umbral de decisión final ya NO se asume en 0.5. Se busca
   explícitamente en el set de VALIDACIÓN con --beta, favoreciendo
   precisión sobre recall (beta < 1) porque el problema reportado es
   demasiados falsos "infestado", no al revés.

Cada corrida genera además una carpeta "reportes/run_<fecha>_<hora>/" con:
    - reporte.txt                 resumen legible de configuración/datos/resultados
    - historial_entrenamiento.csv métricas por época de ambas fases
    - metricas_test.csv           precision/recall/f1/support por clase en test
    - candidatos_revision_manual.csv
    - graficas/*.png               curvas de entrenamiento, matriz de confusión,
                                    curva precisión-recall, distribución de
                                    probabilidades
    - modelos/                     checkpoints .keras y el .tflite final
para poder consultar y comparar corridas después, sin tener que re-entrenar.

Instalar dependencias:
    pip install -r requirements.txt   # incluye matplotlib

Uso típico:
    python entrenamiento_efficientnetb3.py \
        --data_dir ./dataset \
        --manifest manifest.csv \
        --epocas_fase1 10 \
        --epocas_fase2 30
"""

import argparse
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # sin esto, matplotlib intenta abrir una ventana y falla en servidores/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------------------------------------------------
# 0. CONFIGURACIÓN GLOBAL Y REPRODUCIBILIDAD
# ----------------------------------------------------------------------
SEED = 42
IMG_SIZE = 300
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
CLASE_A_INDICE = {"sano": 0, "infestado": 1}

# Un "grupo" = un árbol. Un nombre de archivo tipo "12_infestado_03.jpg"
# o "T045.png" produce el grupo "12" / "T045". Ver advertencia en
# construir_dataframe() sobre cuándo este regex deja de ser confiable.
GROUP_REGEX = re.compile(r"^([A-Za-z]*\d+)")


def fijar_semillas(seed: int = SEED) -> None:
    """Fija la semilla en todas las fuentes de aleatoriedad relevantes
    (Python, NumPy, TensorFlow) para que el split y el entrenamiento
    sean reproducibles entre corridas."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configurar_gpu() -> None:
    """Detecta GPUs disponibles y activa 'memory growth' para no reservar
    toda la VRAM de golpe. En algunos entornos (p. ej. Apple Silicon /
    Metal) esa configuración no es soportada y TensorFlow lanza
    RuntimeError; en ese caso simplemente se informa y se continúa,
    porque en esos entornos el backend ya administra la memoria por su
    cuenta."""
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs detectadas por TensorFlow: {len(gpus)}")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"No se pudo activar memory growth en {gpu.name} ({e}); "
                  f"se continúa con la gestión de memoria por default del backend.")


# ----------------------------------------------------------------------
# 1. CONSTRUCCIÓN DEL DATAFRAME DE TRABAJO
# ----------------------------------------------------------------------
def inferir_grupo(filepath: str) -> str:
    """Extrae el identificador de árbol a partir del nombre de archivo,
    usando GROUP_REGEX. Si no hay match (nombre sin dígitos), se usa el
    nombre completo como grupo -- en la práctica eso equivale a tratar
    esa imagen como su propio grupo aislado."""
    nombre = Path(filepath).stem
    m = GROUP_REGEX.match(nombre)
    return m.group(1) if m else nombre


def construir_dataframe(data_dir: Path, manifest_path: Optional[Path]) -> pd.DataFrame:
    """Construye el DataFrame de trabajo con columnas [filepath, clase, grupo].

    Si existe un manifest.csv válido, se usa como fuente de verdad:
    filtra filas no válidas, quita duplicados exactos por sha256 y,
    si la columna existe, quita duplicados perceptuales (phash_dup).
    Si no hay manifest, se listan directamente las carpetas sano/ e
    infestado/ dentro de data_dir.
    """
    if manifest_path and manifest_path.exists():
        df = pd.read_csv(manifest_path)
        antes = len(df)
        df = df[df["valida"] == True]  # noqa: E712
        df = df.drop_duplicates(subset="sha256")
        if "phash_dup" in df.columns:
            df = df[~df["phash_dup"].fillna(False)]
        print(f"Manifest cargado: {antes} filas -> {len(df)} tras filtrar corruptas/duplicadas.")
        df = df[["filepath", "clase"]].copy()
    else:
        print("No se encontró manifest.csv; leyendo las carpetas directamente.")
        extensiones_validas = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        filas = []
        for clase, nombre_carpeta in {"sano": "sano", "infestado": "infestado"}.items():
            for p in (data_dir / nombre_carpeta).rglob("*"):
                if p.suffix.lower() in extensiones_validas:
                    filas.append({"filepath": str(p), "clase": clase})
        df = pd.DataFrame(filas)

    df["grupo"] = df["filepath"].apply(inferir_grupo)

    # AVISO: si tus imágenes pasaron por un script de copiado/renombrado
    # (p. ej. uno que las renombra por confianza) antes de llegar aquí,
    # es muy probable que el nombre de archivo ya NO conserve el
    # identificador de árbol original (p.ej. "0.930_infestado_001.jpg").
    # En ese caso GROUP_REGEX infiere grupos sin sentido y el split
    # 80/10/10 deja de ser "por árbol": se reintroduce fuga de datos
    # entre train/val/test. Revisa nombres reales de
    # dataset_limpio/sano y dataset_limpio/infestado antes de confiar
    # en este split.
    n_grupos = df["grupo"].nunique()
    print(f"Grupos (árboles) inferidos de los nombres de archivo: {n_grupos}")
    if n_grupos >= len(df) * 0.9:
        print(
            "ADVERTENCIA: casi cada imagen quedó en su propio grupo. Es una señal "
            "de que inferir_grupo() NO está detectando el árbol real -- revisa el "
            "regex o los nombres de archivo antes de entrenar, o el split dejará "
            "de proteger contra fuga de datos entre train/val/test."
        )

    return df


# ----------------------------------------------------------------------
# 2. DIVISIÓN 80/10/10 CONSCIENTE DE GRUPOS
# ----------------------------------------------------------------------
def dividir_dataset(df: pd.DataFrame):
    """Divide el dataset en train/val/test (80/10/10) asegurando que
    todas las imágenes del mismo árbol ('grupo') caigan en el mismo
    split. Esto evita la fuga de datos que ocurriría si dos fotos del
    mismo árbol terminaran una en train y otra en test."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    idx_train, idx_temp = next(gss1.split(df, groups=df["grupo"]))
    df_train, df_temp = df.iloc[idx_train].copy(), df.iloc[idx_temp].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    idx_val, idx_test = next(gss2.split(df_temp, groups=df_temp["grupo"]))
    df_val, df_test = df_temp.iloc[idx_val].copy(), df_temp.iloc[idx_test].copy()

    for nombre, sub in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"{nombre}: {len(sub)} imágenes | {sub['clase'].value_counts().to_dict()}")

    return df_train, df_val, df_test


# ----------------------------------------------------------------------
# 3. PIPELINE tf.data
# ----------------------------------------------------------------------
def cargar_imagen(filepath: tf.Tensor, label: tf.Tensor):
    """Lee un archivo de imagen desde disco, lo decodifica a 3 canales
    y lo redimensiona a IMG_SIZE x IMG_SIZE. No se normaliza a [0,1]
    a propósito: EfficientNet (tf.keras.applications) ya incluye su
    propia capa de normalización interna y espera píxeles en [0,255]."""
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img, label


def construir_tfdataset(df: pd.DataFrame, entrenamiento: bool) -> tf.data.Dataset:
    """Construye un tf.data.Dataset a partir de un DataFrame [filepath, clase].

    - En modo entrenamiento se baraja (shuffle) en cada época.
    - En modo evaluación (val/test) se cachea en memoria tras decodificar,
      porque estos sets se recorren varias veces por época (cálculo de
      métricas propias + búsqueda de umbral) y así se evita releer y
      redecodificar las imágenes cada vez.
    """
    labels = df["clase"].map(CLASE_A_INDICE).values.astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values, labels))
    if entrenamiento:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(cargar_imagen, num_parallel_calls=AUTOTUNE)
    if not entrenamiento:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


# ----------------------------------------------------------------------
# 3.5 CALLBACK: MÉTRICAS DE VALIDACIÓN QUE SÍ TOMAN EN CUENTA A "SANO"
# ----------------------------------------------------------------------
class MetricasValidacionCompletas(tf.keras.callbacks.Callback):
    """
    Al final de cada época calcula la matriz de confusión completa sobre
    el set de validación y agrega al log métricas que consideran ambas
    clases -- no solo el recall de "infestado" que Keras trae por default.

    Por qué: EarlyStopping/ModelCheckpoint monitoreando únicamente
    'val_recall' pueden terminar quedándose con la época donde el modelo
    dice "infestado" casi siempre (recall altísimo, precisión pésima).
    Es exactamente el sesgo reportado en corridas anteriores.

    beta < 1 en el F-beta pondera más la precisión que el recall --
    preferimos, a propósito, un modelo más conservador para declarar
    "infestado".

    Nota: aquí se usa umbral fijo 0.5 para poder comparar épocas entre
    sí de forma consistente durante el entrenamiento. El umbral óptimo
    real se busca aparte, una sola vez, con calcular_umbral_optimo(),
    ya con el modelo final entrenado.
    """

    def __init__(self, ds_val: tf.data.Dataset, y_val: np.ndarray, beta: float = 0.7):
        super().__init__()
        self.ds_val = ds_val
        self.y_val = y_val
        self.beta = beta

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        y_prob = self.model.predict(self.ds_val, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred, labels=[0, 1]).ravel()

        recall_infestado = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall_sano = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # especificidad
        precision_infestado = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        balanced_acc = (recall_infestado + recall_sano) / 2
        if (precision_infestado + recall_infestado) > 0:
            f_beta = (
                (1 + self.beta**2) * precision_infestado * recall_infestado
                / (self.beta**2 * precision_infestado + recall_infestado)
            )
        else:
            f_beta = 0.0

        logs["val_recall_sano"] = recall_sano
        logs["val_balanced_accuracy"] = balanced_acc
        logs["val_fbeta_infestado"] = f_beta

        print(
            f"  val_recall_sano(especificidad)={recall_sano:.3f}  "
            f"val_balanced_accuracy={balanced_acc:.3f}  "
            f"val_fbeta_infestado(beta={self.beta})={f_beta:.3f}  "
            f"[FP={fp} sanos marcados infestado, FN={fn} infestados no detectados]"
        )


# ----------------------------------------------------------------------
# 4. MODELO: AUMENTO DE DATOS + BASE EFFICIENTNET-B3 + HEAD
# ----------------------------------------------------------------------
def construir_modelo(fine_tune_from: Optional[int] = None, dropout: float = 0.3) -> tf.keras.Model:
    """Construye el modelo completo: aumento de datos -> EfficientNet-B3
    (pesos de ImageNet) -> dropout -> capa densa sigmoid de salida.

    fine_tune_from:
        None  -> la base se congela por completo (fase 1, solo se
                 entrena el head).
        int i -> la base se descongela, pero solo desde la capa i en
                 adelante; las capas [:i] permanecen congeladas (fase 2,
                 fine-tuning parcial).

    En ambos casos las capas BatchNormalization se mantienen congeladas
    (trainable=False) incluso durante el fine-tuning, porque sus medias
    y varianzas fueron calibradas sobre ImageNet y actualizarlas con un
    dataset pequeño (~6k imágenes) suele desestabilizar el entrenamiento.
    """
    aumento_de_datos = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.25),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="aumento_de_datos",
    )

    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )

    if fine_tune_from is None:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = aumento_de_datos(inputs)
    x = base_model(x, training=True)
    x = tf.keras.layers.Dropout(dropout, name="dropout_regularizacion")(x)
    # L2 pequeño en la capa final: con un dataset todavía limitado (~6k
    # imágenes en total) es un seguro barato contra sobreajuste.
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="salida_infestado",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)

    return tf.keras.Model(inputs, outputs, name="clasificador_corteza_efficientnetb3")


# ----------------------------------------------------------------------
# 5. PESOS DE CLASE
# ----------------------------------------------------------------------
def calcular_class_weights(df_train: pd.DataFrame, suavizado: float = 0.6) -> dict:
    """
    suavizado=1.0 -> balanceo completo (fórmula 'balanced' de sklearn).
    suavizado=0.0 -> sin ponderar (ambas clases pesan 1.0).

    Por qué atenuar: con el dataset actual (~3,452 sanas vs ~2,634
    infestadas, razón ~1.3:1) el desbalance es leve. El peso "balanced"
    completo ya empuja al modelo a priorizar infestado -- justo la
    dirección del sesgo reportado. Un suavizado entre 0.4 y 0.7 conserva
    algo de compensación sin exagerarla.
    """
    y = df_train["clase"].map(CLASE_A_INDICE).values
    pesos_balanced = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y)
    pesos_finales = 1.0 + suavizado * (pesos_balanced - 1.0)
    class_weight = {0: float(pesos_finales[0]), 1: float(pesos_finales[1])}
    print(f"Pesos de clase (suavizado={suavizado}): {class_weight}")
    return class_weight


# ----------------------------------------------------------------------
# 6. COMPILACIÓN Y CALLBACKS
# ----------------------------------------------------------------------
def compilar(modelo: tf.keras.Model, learning_rate: float) -> None:
    """Compila el modelo con Adam (variante 'legacy', necesaria en
    Apple Silicon para evitar un error de mutación de estado en M1/M2),
    entropía cruzada binaria con label smoothing (regularización extra
    contra sobreconfianza) y las métricas estándar de Keras. Estas
    últimas se complementan en tiempo de entrenamiento con
    MetricasValidacionCompletas, que sí considera ambas clases."""
    modelo.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def callbacks_para(
    nombre_checkpoint: str,
    ds_val: tf.data.Dataset,
    y_val: np.ndarray,
    beta: float = 0.7,
    monitor: str = "val_fbeta_infestado",
):
    """Arma la lista de callbacks para una fase de entrenamiento.

    metricas_cb va PRIMERO en la lista a propósito: Keras pasa el mismo
    dict 'logs' a cada callback en orden dentro de la misma época, así
    que si este callback agrega val_fbeta_infestado antes, EarlyStopping
    y ModelCheckpoint ya lo pueden leer en esa misma época.
    """
    metricas_cb = MetricasValidacionCompletas(ds_val, y_val, beta=beta)
    return [
        metricas_cb,
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            nombre_checkpoint, monitor=monitor, mode="max", save_best_only=True
        ),
    ]


# ----------------------------------------------------------------------
# 7. EVALUACIÓN FINAL
# ----------------------------------------------------------------------
def evaluar(
    modelo: tf.keras.Model,
    ds: tf.data.Dataset,
    df: pd.DataFrame,
    umbral: float = 0.5,
    etiqueta: str = "prueba",
) -> dict:
    """Imprime el reporte de clasificación (precision/recall/F1 por
    clase) y la matriz de confusión de `modelo` sobre `ds`, usando el
    umbral de decisión indicado. Además de imprimir, devuelve un dict
    con los datos crudos (y_true, y_prob, y_pred, reporte, matriz) para
    que quien llame pueda reutilizarlos -- por ejemplo para graficar o
    guardar en el reporte de la corrida -- sin tener que recalcularlos."""
    y_true = df["clase"].map(CLASE_A_INDICE).values
    y_prob = modelo.predict(ds, verbose=0).ravel()
    y_pred = (y_prob >= umbral).astype(int)
    reporte_texto = classification_report(y_true, y_pred, target_names=["sano", "infestado"])
    reporte_dict = classification_report(
        y_true, y_pred, target_names=["sano", "infestado"], output_dict=True
    )
    matriz = confusion_matrix(y_true, y_pred)

    print(f"\n=== Reporte de clasificación (conjunto de {etiqueta}, umbral={umbral:.3f}) ===")
    print(reporte_texto)
    print("Matriz de confusión (filas=real, columnas=predicho):")
    print(matriz)

    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "umbral": umbral,
        "reporte_texto": reporte_texto,
        "reporte_dict": reporte_dict,
        "matriz_confusion": matriz,
    }


# ----------------------------------------------------------------------
# 7.5 BÚSQUEDA DEL UMBRAL ÓPTIMO (NO ASUMIR 0.5)
# ----------------------------------------------------------------------
def calcular_umbral_optimo(
    modelo: tf.keras.Model, ds_val: tf.data.Dataset, y_val: np.ndarray, beta: float = 0.7
) -> dict:
    """
    Busca el umbral que maximiza F-beta en el set de VALIDACIÓN (nunca en
    test, para no contaminar la evaluación final).

    beta < 1 pondera más la precisión que el recall -- preferimos un
    modelo más conservador para declarar "infestado", que es justo el
    comportamiento que se busca corregir. beta=1.0 sería F1 estándar.

    Devuelve un dict con el umbral elegido y, además, la curva
    precisión-recall completa (para poder graficarla después sin volver
    a llamar a modelo.predict).
    """
    y_prob = modelo.predict(ds_val, verbose=0).ravel()
    precisiones, recalls, umbrales = precision_recall_curve(y_val, y_prob)
    # precision_recall_curve regresa un punto extra al final sin umbral asociado
    precisiones, recalls = precisiones[:-1], recalls[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta_scores = (
            (1 + beta**2) * precisiones * recalls / (beta**2 * precisiones + recalls)
        )
    f_beta_scores = np.nan_to_num(f_beta_scores)

    mejor_idx = int(np.argmax(f_beta_scores))
    mejor_umbral = float(umbrales[mejor_idx])

    print(
        f"\nUmbral óptimo en validación (beta={beta}): {mejor_umbral:.3f} "
        f"(precisión={precisiones[mejor_idx]:.3f}, recall={recalls[mejor_idx]:.3f}, "
        f"F-beta={f_beta_scores[mejor_idx]:.3f})"
    )
    print("Como referencia, con umbral=0.5 fijo en el mismo set de validación:")
    y_pred_05 = (y_prob >= 0.5).astype(int)
    print(classification_report(y_val, y_pred_05, target_names=["sano", "infestado"]))

    return {
        "umbral": mejor_umbral,
        "precision": float(precisiones[mejor_idx]),
        "recall": float(recalls[mejor_idx]),
        "f_beta": float(f_beta_scores[mejor_idx]),
        "curva_precision": precisiones,
        "curva_recall": recalls,
        "curva_umbrales": umbrales,
        "y_prob_val": y_prob,
    }


# ----------------------------------------------------------------------
# 8. TRIAGE DE ETIQUETAS DUDOSAS
# ----------------------------------------------------------------------
def triage_etiquetas_dudosas(
    modelo: tf.keras.Model,
    df_train: pd.DataFrame,
    umbral: float = 0.5,
    top_n: int = 150,
    ruta_salida: Path = Path("candidatos_revision_manual.csv"),
) -> None:
    """Corre el modelo sobre el set de entrenamiento y guarda en un CSV
    las `top_n` imágenes donde el modelo discrepa más de la etiqueta
    original, ordenadas por confianza del error. Son las candidatas más
    probables a estar mal etiquetadas y merecen revisión manual.

    ds no se baraja (entrenamiento=False), así que el orden de y_prob
    coincide con el orden de las filas de df_train."""
    ds = construir_tfdataset(df_train, entrenamiento=False)
    y_true = df_train["clase"].map(CLASE_A_INDICE).values
    y_prob = modelo.predict(ds, verbose=0).ravel()

    df_out = df_train.copy()
    df_out["prob_infestado"] = y_prob
    df_out["prediccion"] = (y_prob >= umbral).astype(int)
    df_out["etiqueta_real"] = y_true
    df_out["es_error"] = df_out["prediccion"] != df_out["etiqueta_real"]
    df_out["confianza_del_error"] = np.where(
        df_out["prediccion"] == 1, df_out["prob_infestado"], 1 - df_out["prob_infestado"]
    )

    sospechosas = (
        df_out[df_out["es_error"]].sort_values("confianza_del_error", ascending=False).head(top_n)
    )
    sospechosas.to_csv(ruta_salida, index=False)
    print(f"\n{len(sospechosas)} imágenes marcadas como candidatas a revisión manual.")
    print(f"Guardadas en {ruta_salida}, ordenadas por confianza del error.")


# ----------------------------------------------------------------------
# 9. EXPORTACIÓN A TFLITE
# ----------------------------------------------------------------------
def exportar_tflite(
    modelo: tf.keras.Model, ruta_salida: str, umbral_recomendado: Optional[float] = None
) -> None:
    """Convierte el modelo Keras a TFLite (cuantizado a float16) y lo
    guarda en `ruta_salida`. Si se pasa `umbral_recomendado`, se
    imprime un recordatorio de que el .tflite solo emite la probabilidad
    cruda y que ese umbral debe aplicarse en la app/backend."""
    converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(ruta_salida, "wb") as f:
        f.write(tflite_model)

    tam_mb = len(tflite_model) / (1024 * 1024)
    print(f"\nModelo exportado a {ruta_salida} ({tam_mb:.2f} MB)")
    print("Vuelve a evaluar precisión/recall con el modelo YA cuantizado antes")
    print("de darlo por bueno -- a veces cambia ligeramente respecto al original.")

    if umbral_recomendado is not None:
        print(
            f"\nIMPORTANTE: el .tflite solo emite la probabilidad cruda (sigmoid). "
            f"El umbral óptimo encontrado en validación fue {umbral_recomendado:.3f}, "
            f"NO 0.5. Guarda este número junto con el modelo -- la app/backend debe "
            f"comparar la probabilidad contra este umbral, o el sesgo hacia "
            f"'infestado' puede reaparecer en producción aunque el modelo esté bien "
            f"calibrado."
        )


# ----------------------------------------------------------------------
# 10. REPORTE DE LA CORRIDA (carpeta con txt + csv + gráficas)
# ----------------------------------------------------------------------
def preparar_carpeta_reporte(base_dir: Path = Path("reportes")) -> Path:
    """Crea una carpeta con timestamp dentro de `base_dir` (por default
    'reportes/run_YYYYMMDD_HHMMSS/') donde se van a guardar todos los
    artefactos de esta corrida: modelos, gráficas, CSVs y el reporte en
    texto. Cada corrida queda documentada por separado, para poder
    consultarla o compararla contra otras más adelante."""
    marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta = base_dir / f"run_{marca_tiempo}"
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def construir_df_historial(historial_fase1, historial_fase2) -> pd.DataFrame:
    """Une los objetos History devueltos por model.fit() de ambas fases
    en un único DataFrame de largo formato (una fila por época), con
    una columna 'fase' y una 'epoca_global' continua entre fase 1 y
    fase 2. Como MetricasValidacionCompletas corre antes de que Keras
    registre el History interno, las columnas val_recall_sano,
    val_balanced_accuracy y val_fbeta_infestado ya están incluidas en
    history.history sin trabajo extra."""
    filas = []
    for fase, historial in [("fase1", historial_fase1), ("fase2", historial_fase2)]:
        n_epocas = len(next(iter(historial.history.values())))
        for i in range(n_epocas):
            fila = {"fase": fase, "epoca_en_fase": i + 1}
            for metrica, valores in historial.history.items():
                fila[metrica] = valores[i]
            filas.append(fila)
    df_historial = pd.DataFrame(filas)
    df_historial["epoca_global"] = range(1, len(df_historial) + 1)
    return df_historial


def graficar_curvas_entrenamiento(df_historial: pd.DataFrame, ruta_salida: Path) -> None:
    """Grafica pérdida, exactitud, AUC y precisión (train vs. val) a lo
    largo de ambas fases, con una línea vertical marcando dónde empieza
    el fine-tuning (fase 2)."""
    frontera_fase2 = (df_historial["fase"] == "fase1").sum() + 0.5
    paneles = [
        ("loss", "val_loss", "Pérdida (loss)"),
        ("accuracy", "val_accuracy", "Exactitud (accuracy)"),
        ("auc", "val_auc", "AUC"),
        ("precision", "val_precision", "Precisión (umbral=0.5)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (m_train, m_val, titulo) in zip(axes.ravel(), paneles):
        if m_train in df_historial.columns:
            ax.plot(df_historial["epoca_global"], df_historial[m_train], label="entrenamiento")
        if m_val in df_historial.columns:
            ax.plot(df_historial["epoca_global"], df_historial[m_val], label="validación")
        ax.axvline(frontera_fase2, color="gray", linestyle="--", linewidth=1)
        ax.set_title(titulo)
        ax.set_xlabel("Época")
        ax.legend(fontsize=8)
    fig.suptitle("Curvas de entrenamiento (fase 1: head | fase 2: fine-tuning)")
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)


def graficar_metricas_balanceadas(df_historial: pd.DataFrame, ruta_salida: Path) -> None:
    """Grafica la evolución de las métricas que sí consideran ambas
    clases (recall de sano, balanced accuracy, F-beta de infestado) --
    las que motivaron el cambio de qué se monitorea para elegir
    checkpoint."""
    frontera_fase2 = (df_historial["fase"] == "fase1").sum() + 0.5
    fig, ax = plt.subplots(figsize=(9, 5))
    for columna, etiqueta in [
        ("val_recall_sano", "recall sano (especificidad)"),
        ("val_balanced_accuracy", "balanced accuracy"),
        ("val_fbeta_infestado", "F-beta infestado"),
    ]:
        if columna in df_historial.columns:
            ax.plot(df_historial["epoca_global"], df_historial[columna], label=etiqueta)
    ax.axvline(frontera_fase2, color="gray", linestyle="--", linewidth=1, label="inicio fase 2")
    ax.set_xlabel("Época")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.02)
    ax.set_title("Métricas de validación que consideran ambas clases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)


def graficar_matriz_confusion(matriz: np.ndarray, ruta_salida: Path, etiqueta: str = "prueba") -> None:
    """Dibuja la matriz de confusión (sano/infestado) como mapa de calor
    con las cuentas anotadas en cada celda."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matriz, cmap="Blues")
    clases = ["sano", "infestado"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(clases)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(clases)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión ({etiqueta})")
    vmax = matriz.max() if matriz.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            color = "white" if matriz[i, j] > vmax / 2 else "black"
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center", color=color, fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)


def graficar_curva_precision_recall(info_umbral: dict, beta: float, ruta_salida: Path) -> None:
    """Grafica la curva precisión-recall calculada en validación y marca
    el punto correspondiente al umbral finalmente elegido."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(info_umbral["curva_recall"], info_umbral["curva_precision"], label="curva P-R (validación)")
    ax.scatter(
        [info_umbral["recall"]],
        [info_umbral["precision"]],
        color="red",
        zorder=5,
        label=f"umbral elegido = {info_umbral['umbral']:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_title(f"Precisión vs. Recall en validación (beta={beta})")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)


def graficar_distribucion_probabilidades(
    y_true: np.ndarray, y_prob: np.ndarray, umbral: float, ruta_salida: Path, etiqueta: str = "prueba"
) -> None:
    """Histograma de las probabilidades predichas, separado por clase
    real, con el umbral usado marcado como línea vertical. Ayuda a ver
    qué tan bien separa el modelo ambas clases y cuántos casos quedan
    justo alrededor del umbral (los más ambiguos)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y_prob[y_true == 0], bins=25, alpha=0.6, range=(0, 1), label="sano (real)")
    ax.hist(y_prob[y_true == 1], bins=25, alpha=0.6, range=(0, 1), label="infestado (real)")
    ax.axvline(umbral, color="red", linestyle="--", label=f"umbral usado = {umbral:.3f}")
    ax.set_xlabel("Probabilidad predicha de infestado")
    ax.set_ylabel("Número de imágenes")
    ax.set_title(f"Distribución de probabilidades predichas ({etiqueta})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)


def generar_reporte_txt(
    carpeta_reporte: Path,
    args: argparse.Namespace,
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    class_weight: dict,
    df_historial: pd.DataFrame,
    info_umbral: dict,
    resultado_test: dict,
    ruta_tflite: Path,
    tam_tflite_mb: float,
    archivos_generados: list,
) -> Path:
    """Arma reporte.txt: un resumen legible de configuración, datos,
    entrenamiento, umbral elegido y evaluación final en test, más un
    índice de todos los archivos que quedaron en la carpeta de esta
    corrida."""
    epocas_fase1_reales = int((df_historial["fase"] == "fase1").sum())
    epocas_fase2_reales = int((df_historial["fase"] == "fase2").sum())

    lineas = [
        "REPORTE DE ENTRENAMIENTO - Clasificador de corteza (sano / infestado)",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "\n1. CONFIGURACIÓN",
        f"  data_dir: {args.data_dir}",
        f"  manifest: {args.manifest}",
        f"  img_size: {IMG_SIZE}  batch_size: {BATCH_SIZE}  seed: {SEED}",
        f"  epocas_fase1 (planeadas): {args.epocas_fase1}  |  corridas: {epocas_fase1_reales}",
        f"  epocas_fase2 (planeadas): {args.epocas_fase2}  |  corridas: {epocas_fase2_reales}",
        f"  descongelar_desde: {args.descongelar_desde}",
        f"  suavizado_pesos: {args.suavizado_pesos}",
        f"  beta (F-beta): {args.beta}",
        f"  dropout: {args.dropout}",
        "\n2. DATOS",
        f"  Total imágenes tras filtrado: {len(df)}",
        f"  Árboles (grupos) detectados: {df['grupo'].nunique()}",
    ]
    for nombre, sub in [("train", df_train), ("val", df_val), ("test", df_test)]:
        lineas.append(f"  {nombre}: {len(sub)} imágenes | {sub['clase'].value_counts().to_dict()}")
    lineas.append(f"  Pesos de clase usados: {class_weight}")

    lineas += [
        "\n3. UMBRAL ÓPTIMO (buscado en validación, nunca en test)",
        f"  Umbral elegido: {info_umbral['umbral']:.4f}",
        f"  En ese umbral (validación): precisión={info_umbral['precision']:.3f}, "
        f"recall={info_umbral['recall']:.3f}, F-beta={info_umbral['f_beta']:.3f}",
        "\n4. EVALUACIÓN FINAL EN TEST (con umbral óptimo)",
        resultado_test["reporte_texto"],
        "Matriz de confusión (filas=real, columnas=predicho):",
        str(resultado_test["matriz_confusion"]),
        "\n5. EXPORTACIÓN",
        f"  Modelo TFLite: {ruta_tflite} ({tam_tflite_mb:.2f} MB)",
        f"  Umbral a aplicar en producción: {info_umbral['umbral']:.4f} (NO usar 0.5)",
        "\n6. ARCHIVOS GENERADOS EN ESTA CARPETA",
    ]
    lineas += [f"  - {f}" for f in archivos_generados]

    contenido = "\n".join(lineas)
    ruta_txt = carpeta_reporte / "reporte.txt"
    ruta_txt.write_text(contenido, encoding="utf-8")
    return ruta_txt


# ----------------------------------------------------------------------
# 11. ARGUMENTOS DE LÍNEA DE COMANDOS
# ----------------------------------------------------------------------
def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("dataset"))
    parser.add_argument("--manifest", type=Path, default=Path("manifest.csv"))
    parser.add_argument("--epocas_fase1", type=int, default=10)
    parser.add_argument("--epocas_fase2", type=int, default=30)
    parser.add_argument(
        "--descongelar_desde",
        type=int,
        default=250,
        help=(
            "EfficientNet-B3 tiene varios cientos de capas (cerca de 384). "
            "Con un dataset todavía limitado (~6k imágenes en total), "
            "descongelar menos capas (número más alto) reduce el riesgo de "
            "sobreajuste comparado con descongelar desde una capa muy temprana."
        ),
    )
    parser.add_argument(
        "--suavizado_pesos",
        type=float,
        default=0.6,
        help="0.0 = sin ponderar clases, 1.0 = balanceo completo de sklearn. Ver calcular_class_weights().",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.7,
        help=(
            "Controla el F-beta usado para elegir checkpoint y umbral final. "
            "beta < 1 favorece precisión (menos falsos 'infestado'); "
            "beta = 1 es F1 estándar; beta > 1 favorece recall."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    return parser.parse_args()


# ----------------------------------------------------------------------
# 12. FLUJO PRINCIPAL
# ----------------------------------------------------------------------
def main() -> None:
    fijar_semillas()
    configurar_gpu()

    args = parsear_argumentos()

    # Carpeta donde queda documentada esta corrida: modelos, gráficas,
    # CSVs y el reporte.txt final. Se crea al inicio para poder guardar
    # ahí los checkpoints a medida que se entrena.
    carpeta_reporte = preparar_carpeta_reporte()
    carpeta_modelos = carpeta_reporte / "modelos"
    carpeta_graficas = carpeta_reporte / "graficas"
    carpeta_modelos.mkdir(parents=True, exist_ok=True)
    carpeta_graficas.mkdir(parents=True, exist_ok=True)
    print(f"\nEsta corrida se documentará en: {carpeta_reporte}")

    df = construir_dataframe(args.data_dir, args.manifest)
    df_train, df_val, df_test = dividir_dataset(df)

    ds_train = construir_tfdataset(df_train, entrenamiento=True)
    ds_val = construir_tfdataset(df_val, entrenamiento=False)
    ds_test = construir_tfdataset(df_test, entrenamiento=False)
    y_val = df_val["clase"].map(CLASE_A_INDICE).values.astype("int32")

    class_weight = calcular_class_weights(df_train, suavizado=args.suavizado_pesos)

    print("\n=== FASE 1: entrenando solo el head (base congelada) ===")
    modelo = construir_modelo(fine_tune_from=None, dropout=args.dropout)
    compilar(modelo, learning_rate=1e-3)
    historial_fase1 = modelo.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epocas_fase1,
        class_weight=class_weight,
        callbacks=callbacks_para(
            str(carpeta_modelos / "mejor_modelo_fase1.keras"), ds_val, y_val, beta=args.beta
        ),
    )

    print("\n--- Chequeo rápido tras fase 1 (antes de fine-tuning) ---")
    evaluar(modelo, ds_val, df_val, umbral=0.5, etiqueta="validación")

    print("\n=== FASE 2: fine-tuning con la base parcialmente descongelada ===")
    modelo_ft = construir_modelo(fine_tune_from=args.descongelar_desde, dropout=args.dropout)
    modelo_ft.set_weights(modelo.get_weights())
    compilar(modelo_ft, learning_rate=1e-5)
    historial_fase2 = modelo_ft.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epocas_fase2,
        class_weight=class_weight,
        callbacks=callbacks_para(
            str(carpeta_modelos / "mejor_modelo_fase2.keras"), ds_val, y_val, beta=args.beta
        ),
    )

    info_umbral = calcular_umbral_optimo(modelo_ft, ds_val, y_val, beta=args.beta)
    umbral_optimo = info_umbral["umbral"]

    resultado_test = evaluar(modelo_ft, ds_test, df_test, umbral=umbral_optimo, etiqueta="prueba")
    triage_etiquetas_dudosas(
        modelo_ft,
        df_train,
        umbral=umbral_optimo,
        ruta_salida=carpeta_reporte / "candidatos_revision_manual.csv",
    )

    ruta_tflite = carpeta_modelos / "bark_classifier.tflite"
    exportar_tflite(modelo_ft, str(ruta_tflite), umbral_recomendado=umbral_optimo)
    tam_tflite_mb = ruta_tflite.stat().st_size / (1024 * 1024)

    # --- Reporte de la corrida: CSVs + gráficas + resumen en texto ---
    df_historial = construir_df_historial(historial_fase1, historial_fase2)
    df_historial.to_csv(carpeta_reporte / "historial_entrenamiento.csv", index=False)
    pd.DataFrame(resultado_test["reporte_dict"]).transpose().to_csv(
        carpeta_reporte / "metricas_test.csv"
    )

    graficar_curvas_entrenamiento(df_historial, carpeta_graficas / "curvas_entrenamiento.png")
    graficar_metricas_balanceadas(df_historial, carpeta_graficas / "metricas_balanceadas.png")
    graficar_matriz_confusion(
        resultado_test["matriz_confusion"], carpeta_graficas / "matriz_confusion_test.png", etiqueta="prueba"
    )
    graficar_curva_precision_recall(info_umbral, args.beta, carpeta_graficas / "curva_precision_recall.png")
    graficar_distribucion_probabilidades(
        resultado_test["y_true"],
        resultado_test["y_prob"],
        umbral_optimo,
        carpeta_graficas / "distribucion_probabilidades_test.png",
        etiqueta="prueba",
    )

    archivos_generados = [
        "reporte.txt",
        "historial_entrenamiento.csv",
        "metricas_test.csv",
        "candidatos_revision_manual.csv",
        "graficas/curvas_entrenamiento.png",
        "graficas/metricas_balanceadas.png",
        "graficas/matriz_confusion_test.png",
        "graficas/curva_precision_recall.png",
        "graficas/distribucion_probabilidades_test.png",
        "modelos/mejor_modelo_fase1.keras",
        "modelos/mejor_modelo_fase2.keras",
        "modelos/bark_classifier.tflite",
    ]
    ruta_txt = generar_reporte_txt(
        carpeta_reporte,
        args,
        df,
        df_train,
        df_val,
        df_test,
        class_weight,
        df_historial,
        info_umbral,
        resultado_test,
        ruta_tflite,
        tam_tflite_mb,
        archivos_generados,
    )

    print(f"\nReporte completo guardado en: {carpeta_reporte}")
    print(f"  -> {ruta_txt}")


if __name__ == "__main__":
    main()
