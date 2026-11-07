"""
detectar_duplicados_carpeta.py

Detecta imagenes duplicadas DENTRO de una sola carpeta -- pensado para
revisar dataset_limpio/sano (o infestado) despues de copiarlas con
copiar_troncos_por_confianza.py.

Compara en tres niveles, de mas debil a mas confiable:

1. Tamano en bytes: rapido, pero NO concluyente por si solo. Dos fotos
   distintas pueden coincidir en tamano por casualidad, y la MISMA
   foto guardada con otra compresion puede pesar distinto. Se usa solo
   como primer vistazo, no como prueba.
2. Hash exacto de contenido (SHA-256): si dos archivos comparten
   hash, son byte-por-byte identicos. Esto SI es concluyente.
3. Hash perceptual (phash): detecta casi-duplicados -- la misma foto
   recomprimida, redimensionada o en otro formato, que ya no comparte
   el hash exacto pero visualmente es la misma imagen.

Como tus archivos ya vienen renombrados con el prefijo de rango de
confianza (00001_conf0.998_...), dentro de cada grupo duplicado el
script recomienda conservar el de menor numero de rango (mayor
confianza) y marca el resto para revisar/mover.

Uso -- solo reportar, sin tocar nada:
    python detectar_duplicados_carpeta.py --carpeta dataset_limpio/sano

Uso -- mover los duplicados sobrantes a otra carpeta (no los borra,
los deja ahi por si quieres confirmar antes de eliminarlos de verdad):
    python detectar_duplicados_carpeta.py --carpeta dataset_limpio/sano --mover_a duplicados_revisar
"""

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
import imagehash

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def hash_exacto(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_perceptual(path: Path):
    try:
        with Image.open(path) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None


def marcar_conservar(df: pd.DataFrame, columna_grupo: str) -> set:
    """De cada grupo duplicado, conserva el de nombre 'menor' -- como
    los nombres empiezan con el rango de confianza (00001, 00002...),
    el menor nombre = mayor confianza."""
    conservar = set()
    validos = df[df[columna_grupo].notna()]
    for _, grupo in validos.groupby(columna_grupo):
        if len(grupo) > 1:
            mejor = grupo.sort_values("nombre").iloc[0]["filepath"]
            conservar.add(mejor)
    return conservar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carpeta", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("duplicados_detectados.csv"))
    parser.add_argument(
        "--mover_a",
        type=Path,
        default=None,
        help="Si se especifica, mueve (no borra) los duplicados sobrantes a esta carpeta.",
    )
    args = parser.parse_args()

    archivos = [p for p in sorted(args.carpeta.iterdir()) if p.suffix.lower() in VALID_EXTENSIONS]
    print(f"Revisando {len(archivos)} imagenes en {args.carpeta}...")

    df = pd.DataFrame([
        {"filepath": str(p), "nombre": p.name, "tamano_bytes": p.stat().st_size}
        for p in archivos
    ])

    # --- Nivel 1: mismo tamano en bytes (solo para tener una idea rapida) ---
    mismo_tamano = df[df.duplicated("tamano_bytes", keep=False)]
    print(f"\nArchivos que comparten tamano exacto en bytes con al menos otro: {len(mismo_tamano)}")
    print("(no concluyente por si solo -- se confirma con el hash de abajo)")

    # --- Nivel 2: hash exacto de contenido ---
    print("Calculando hash exacto (SHA-256)...")
    df["sha256"] = df["filepath"].apply(lambda p: hash_exacto(Path(p)))
    duplicados_exactos = df[df.duplicated("sha256", keep=False)]
    print(f"Duplicados EXACTOS (contenido identico byte por byte): {len(duplicados_exactos)}")

    # --- Nivel 3: hash perceptual (casi-duplicados) ---
    print("Calculando hash perceptual (phash)...")
    df["phash"] = df["filepath"].apply(lambda p: hash_perceptual(Path(p)))
    ya_exactos = set(duplicados_exactos["filepath"])
    casi_duplicados = df[
        df.duplicated("phash", keep=False) & df["phash"].notna() & ~df["filepath"].isin(ya_exactos)
    ]
    print(f"Posibles CASI-duplicados (misma foto, distinta compresion/tamano): {len(casi_duplicados)}")

    conservar_exactos = marcar_conservar(df, "sha256")
    conservar_phash = marcar_conservar(df, "phash")

    df["duplicado_exacto"] = df["filepath"].isin(duplicados_exactos["filepath"])
    df["casi_duplicado"] = df["filepath"].isin(casi_duplicados["filepath"])
    df["conservar"] = ~(
        (df["duplicado_exacto"] & ~df["filepath"].isin(conservar_exactos))
        | (df["casi_duplicado"] & ~df["filepath"].isin(conservar_phash))
    )

    df.to_csv(args.out, index=False)
    n_borrar = (~df["conservar"]).sum()
    print(f"\nReporte guardado en {args.out}")
    print(f"Recomendado quitar: {n_borrar} imagenes (se conserva la de mayor confianza de cada grupo duplicado)")

    if args.mover_a and n_borrar:
        args.mover_a.mkdir(parents=True, exist_ok=True)
        for fp in df[~df["conservar"]]["filepath"]:
            shutil.move(fp, args.mover_a / Path(fp).name)
        print(f"Movidas {n_borrar} imagenes a {args.mover_a} (no se borraron, solo se sacaron de la carpeta principal).")


if __name__ == "__main__":
    main()
