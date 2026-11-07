"""
copiar_troncos_por_confianza.py

Copia las imagenes clasificadas como "tronco" a una carpeta nueva,
renombradas con un prefijo numerico segun su confianza -- asi, al
ordenar alfabeticamente en Finder (o donde sea), las primeras son las
que el modelo cree con MAS seguridad que son tronco, y las ultimas las
mas dudosas. Util para revisar a ojo si la calidad se degrada hacia el
final del rango.

Uso:
    python copiar_troncos_por_confianza.py --entrada tipos_final_binario.csv --destino dataset_limpio/sano
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=Path("tipos_final_binario.csv"))
    parser.add_argument("--destino", type=Path, default=Path("dataset_limpio/sano"))
    parser.add_argument(
        "--incluir_revisar_manual",
        action="store_true",
        help="Si se activa, tambien copia las marcadas revisar_manual=True (al final del rango, con menor confianza).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.entrada)
    df = df[df["tipo_predicho"] == "tronco"]
    if not args.incluir_revisar_manual:
        df = df[~df["revisar_manual"]]

    # Orden descendente por confianza: la #00001 es la mas segura.
    df = df.sort_values("confianza", ascending=False).reset_index(drop=True)

    args.destino.mkdir(parents=True, exist_ok=True)

    for rango, fila in df.iterrows():
        origen = Path(fila["filepath"])
        nombre_nuevo = f"{rango + 1:05d}_conf{fila['confianza']:.3f}_{origen.name}"
        shutil.copy(origen, args.destino / nombre_nuevo)

    print(f"{len(df)} imagenes copiadas a {args.destino}, ordenadas por confianza descendente.")
    if len(df):
        print(f"Confianza mas alta: {df['confianza'].iloc[0]:.3f}")
        print(f"Confianza mas baja incluida: {df['confianza'].iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
