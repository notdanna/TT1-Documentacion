"""
diagnosticar_carpeta.py

Lee los nombres de archivo ya generados por copiar_troncos_por_confianza.py
(formato: 00001_conf0.998_nombreoriginal.jpg) y diagnostica por que una
carpeta tiene mas imagenes de las esperadas, SIN necesidad de volver a
calcular hashes.

Revisa dos cosas:

1. Si el numero de rango (los primeros 5 digitos) se repite -- eso
   confirma que corriste el script de copiado mas de una vez sin
   vaciar la carpeta antes, porque cada corrida reinicia el conteo
   desde 00001.
2. La distribucion de confianza -- si hay muchas imagenes con
   confianza menor a 0.5 en una carpeta que se supone son puro
   "tronco", eso sugiere que se copio el conjunto equivocado (o con
   --incluir_revisar_manual).

Uso:
    python diagnosticar_carpeta.py --carpeta dataset_limpio/sano
"""

import argparse
import re
from pathlib import Path

import pandas as pd

PATRON = re.compile(r"^(\d{5})_conf([0-9.]+)_(.+)$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carpeta", type=Path, required=True)
    args = parser.parse_args()

    archivos = sorted(args.carpeta.iterdir())
    print(f"Total de archivos en {args.carpeta}: {len(archivos)}")

    filas = []
    sin_patron = 0
    for p in archivos:
        m = PATRON.match(p.name)
        if not m:
            sin_patron += 1
            continue
        rango, confianza, resto = m.groups()
        filas.append({"archivo": p.name, "rango": rango, "confianza": float(confianza), "resto": resto})

    if sin_patron:
        print(f"Aviso: {sin_patron} archivos no tienen el formato esperado (rango_conf_nombre) y se ignoran en el analisis.")

    df = pd.DataFrame(filas)
    if df.empty:
        print("Ningun archivo coincide con el formato esperado -- revisa que sea la carpeta correcta.")
        return

    # --- Diagnostico 1: rangos repetidos = multiples corridas encimadas ---
    rangos_repetidos = df[df.duplicated("rango", keep=False)].sort_values("rango")
    print(f"\nNumeros de rango que se repiten (evidencia de corridas encimadas): {rangos_repetidos['rango'].nunique()}")
    if len(rangos_repetidos):
        print("Ejemplos:")
        print(rangos_repetidos.head(10)[["rango", "archivo"]].to_string(index=False))
        print("\n-> Si ves esto, corriste copiar_troncos_por_confianza.py mas de una vez")
        print("   sin vaciar la carpeta antes. Borra la carpeta y vuelve a copiar UNA SOLA VEZ")
        print("   desde el CSV final (tipos_final_binario.csv).")

    # --- Diagnostico 2: distribucion de confianza ---
    print(f"\nConfianza minima: {df['confianza'].min():.3f}")
    print(f"Confianza maxima: {df['confianza'].max():.3f}")
    print(f"Confianza promedio: {df['confianza'].mean():.3f}")

    baja_confianza = df[df["confianza"] < 0.5]
    print(f"\nImagenes con confianza menor a 0.5: {len(baja_confianza)}")
    if len(baja_confianza) > 0.05 * len(df):
        print("-> Esto es sospechoso para una carpeta que deberia ser puro 'tronco' con alta")
        print("   confianza. Revisa si copiaste el CSV/columna correcta, o si usaste")
        print("   --incluir_revisar_manual sin querer.")


if __name__ == "__main__":
    main()
