"""
revisar_muestra.py

Herramienta interactiva para validar a ojo la muestra generada por
clip_filtro_tipo_imagen.py, SIN tener que editar un CSV a mano.

Muestra una imagen a la vez en una ventana; tu solo presionas una
tecla:

    ENTER  -> la prediccion de CLIP esta correcta, se confirma tal cual
    t      -> en realidad es "tronco"
    f      -> en realidad es "follaje"
    a      -> en realidad es "arbol_completo"
    o      -> en realidad es "otro"
    q      -> guardar el progreso y salir (puedes retomar despues)

Guarda el resultado despues de CADA imagen (no al final), asi que si
cierras la ventana o hay un error, no pierdes lo ya revisado. Si vuelves
a correr el script, retoma donde te quedaste automaticamente.

Instalar dependencia adicional:
    pip install matplotlib

Uso:
    python revisar_muestra.py --muestra muestra_para_corregir.csv --salida muestra_corregida.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

TECLAS_A_CATEGORIA = {
    "t": "tronco",
    "f": "follaje",
    "a": "arbol_completo",
    "o": "otro",
}


def revisar(df: pd.DataFrame, salida_path: Path):
    # Si ya existe un progreso previo, no vuelve a mostrar las imagenes
    # ya revisadas -- asi puedes cerrar y retomar cuando quieras.
    resultados = []
    if salida_path.exists():
        previos = pd.read_csv(salida_path)
        resultados = previos.to_dict("records")
        ya_revisadas = set(previos["filepath"])
        df = df[~df["filepath"].isin(ya_revisadas)].reset_index(drop=True)
        print(f"Retomando: {len(previos)} imagenes ya revisadas antes, quedan {len(df)}.")

    if len(df) == 0:
        print("No quedan imagenes por revisar en esta muestra.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    estado = {"tecla": None}

    def al_presionar_tecla(event):
        estado["tecla"] = event.key

    fig.canvas.mpl_connect("key_press_event", al_presionar_tecla)

    total = len(df)
    for i, fila in df.iterrows():
        try:
            img = Image.open(fila["filepath"]).convert("RGB")
        except Exception as exc:
            print(f"No se pudo abrir {fila['filepath']}: {exc} -- se omite.")
            continue

        ax.clear()
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"[{i + 1}/{total}]  CLIP predijo: {fila['tipo_predicho']}  "
            f"(confianza {fila['confianza']:.2f}, margen {fila['margen']:.2f})\n"
            "ENTER = correcto   t=tronco  f=follaje  a=arbol_completo  o=otro   q = guardar y salir",
            fontsize=9,
        )
        plt.draw()

        estado["tecla"] = None
        while estado["tecla"] is None:
            plt.waitforbuttonpress(timeout=0.5)
        tecla = estado["tecla"]

        if tecla == "q":
            print("\nGuardando progreso y saliendo...")
            break
        elif tecla == "enter":
            tipo_final = fila["tipo_predicho"]
        elif tecla in TECLAS_A_CATEGORIA:
            tipo_final = TECLAS_A_CATEGORIA[tecla]
        else:
            print(f"Tecla '{tecla}' no reconocida, se conserva la prediccion de CLIP.")
            tipo_final = fila["tipo_predicho"]

        resultados.append({"filepath": fila["filepath"], "tipo": tipo_final})
        # Guarda despues de CADA imagen -- si algo falla a la mitad, no se pierde nada.
        pd.DataFrame(resultados).to_csv(salida_path, index=False)

    plt.close(fig)
    print(f"\nListo. {len(resultados)} imagenes con etiqueta confirmada en: {salida_path}")
    print("Ese archivo ya tiene el formato filepath,tipo listo para --seed_labels.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muestra", type=Path, default=Path("muestra_para_corregir.csv"))
    parser.add_argument("--salida", type=Path, default=Path("muestra_corregida.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.muestra)
    revisar(df, args.salida)


if __name__ == "__main__":
    main()
