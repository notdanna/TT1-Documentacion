"""
entrenamiento_efficientnetb3.py

Entrena un clasificador binario de corteza (sano / infestado) con
transfer learning en dos fases sobre EfficientNet-B3.

1. EarlyStopping/ModelCheckpoint ya NO monitorean solo val_recall.
   val_recall en un problema binario con sigmoid mide únicamente el
   recall de la clase 1 (infestado) e ignora por completo cuántos
   sanos se están clasificando mal. Un modelo que dice "infestado"
   casi siempre tiene val_recall altísimo y aun así es un mal modelo. => OVERFITTING
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

Instalar dependencias:
    pip install -r requirements.txt

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
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

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


# CONSTRUCCIÓN DEL DATAFRAME DE TRABAJO
def inferir_grupo(filepath: str) -> str:
    """Extrae el identificador de árbol a partir del nombre de archivo,
    usando GROUP_REGEX. Si no hay match (nombre sin dígitos), se usa el
    nombre completo como grupo -- en la práctica eso equivale a tratar
    esa imagen como su propio grupo aislado. LO CUAL ES MALO """
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


# DIVISIÓN 80/10/10 CONSCIENTE DE GRUPOS
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


# PIPELINE tf.data
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
) -> None:
    """Imprime el reporte de clasificación (precision/recall/F1 por
    clase) y la matriz de confusión de `modelo` sobre `ds`, usando el
    umbral de decisión indicado."""
    y_true = df["clase"].map(CLASE_A_INDICE).values
    y_prob = modelo.predict(ds, verbose=0).ravel()
    y_pred = (y_prob >= umbral).astype(int)
    print(f"\n=== Reporte de clasificación (conjunto de {etiqueta}, umbral={umbral:.3f}) ===")
    print(classification_report(y_true, y_pred, target_names=["sano", "infestado"]))
    print("Matriz de confusión (filas=real, columnas=predicho):")
    print(confusion_matrix(y_true, y_pred))


# ----------------------------------------------------------------------
# 7.5 BÚSQUEDA DEL UMBRAL ÓPTIMO (NO ASUMIR 0.5)
# ----------------------------------------------------------------------
def calcular_umbral_optimo(
    modelo: tf.keras.Model, ds_val: tf.data.Dataset, y_val: np.ndarray, beta: float = 0.7
) -> float:
    """
    Busca el umbral que maximiza F-beta en el set de VALIDACIÓN (nunca en
    test, para no contaminar la evaluación final).

    beta < 1 pondera más la precisión que el recall -- preferimos un
    modelo más conservador para declarar "infestado", que es justo el
    comportamiento que se busca corregir. beta=1.0 sería F1 estándar.
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

    return mejor_umbral


# ----------------------------------------------------------------------
# 8. TRIAGE DE ETIQUETAS DUDOSAS
# ----------------------------------------------------------------------
def triage_etiquetas_dudosas(
    modelo: tf.keras.Model, df_train: pd.DataFrame, umbral: float = 0.5, top_n: int = 150
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
    sospechosas.to_csv("candidatos_revision_manual.csv", index=False)
    print(f"\n{len(sospechosas)} imágenes marcadas como candidatas a revisión manual.")
    print("Guardadas en candidatos_revision_manual.csv, ordenadas por confianza del error.")


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
# 10. ARGUMENTOS DE LÍNEA DE COMANDOS
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
# 11. FLUJO PRINCIPAL
# ----------------------------------------------------------------------
def main() -> None:
    fijar_semillas()
    configurar_gpu()

    args = parsear_argumentos()

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
    modelo.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epocas_fase1,
        class_weight=class_weight,
        callbacks=callbacks_para("mejor_modelo_fase1.keras", ds_val, y_val, beta=args.beta),
    )

    print("\n--- Chequeo rápido tras fase 1 (antes de fine-tuning) ---")
    evaluar(modelo, ds_val, df_val, umbral=0.5, etiqueta="validación")

    print("\n=== FASE 2: fine-tuning con la base parcialmente descongelada ===")
    modelo_ft = construir_modelo(fine_tune_from=args.descongelar_desde, dropout=args.dropout)
    modelo_ft.set_weights(modelo.get_weights())
    compilar(modelo_ft, learning_rate=1e-5)
    modelo_ft.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epocas_fase2,
        class_weight=class_weight,
        callbacks=callbacks_para("mejor_modelo_fase2.keras", ds_val, y_val, beta=args.beta),
    )

    umbral_optimo = calcular_umbral_optimo(modelo_ft, ds_val, y_val, beta=args.beta)
    evaluar(modelo_ft, ds_test, df_test, umbral=umbral_optimo, etiqueta="prueba")
    triage_etiquetas_dudosas(modelo_ft, df_train, umbral=umbral_optimo)
    exportar_tflite(modelo_ft, "bark_classifier.tflite", umbral_recomendado=umbral_optimo)


if __name__ == "__main__":
    main()
