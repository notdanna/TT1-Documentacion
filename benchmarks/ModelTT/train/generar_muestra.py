"""
generar_muestra.py

Toma tipos_zero_shot.csv (o el CSV que le indiques) y saca una muestra
estratificada por categoria predicha, para revisar a ojo con
revisar_muestra.py en vez de las miles de imagenes completas.

Uso:
    python generar_muestra.py --entrada tipos_zero_shot.csv --salida muestra_para_corregir.csv --n_por_categoria 50
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=Path("tipos_zero_shot.csv"))
    parser.add_argument("--salida", type=Path, default=Path("muestra_para_corregir.csv"))
    parser.add_argument("--n_por_categoria", type=int, default=50)
    args = parser.parse_args()

    if not args.entrada.exists():
        print(f"ERROR: no encuentro '{args.entrada}' en {Path.cwd()}")
        print("Revisa que estes parado en la carpeta correcta (usa 'pwd' y 'ls *.csv').")
        return

    df = pd.read_csv(args.entrada)
    df_validas = df[df["imagen_valida"] == True]  # noqa: E712

    muestra = (
        df_validas.groupby("tipo_predicho", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), args.n_por_categoria), random_state=42))
    )

    muestra.to_csv(args.salida, index=False)
    print(f"Muestra guardada en {args.salida} ({len(muestra)} imagenes)")
    print(muestra["tipo_predicho"].value_counts().to_string())


if __name__ == "__main__":
    main()
