"""
matrix_viz — visualizaciones de Matrices (arrays 2D) para el notebook de la lección.

Todas las funciones reciben un numpy.ndarray 2-D como entrada y dibujan el
resultado en la salida de la celda con IPython.display.HTML.

Cubre las operaciones de la lección:
    memoria · get · set · recorrido (traverse) · inserción/eliminación de fila
    · búsqueda lineal · búsqueda binaria

Concepto central — ORDEN FILA-MAYOR (row-major, el que usa C y numpy):
    Una matriz vive en memoria como un array 1-D contiguo, fila tras fila.
    El elemento (i, j) de una matriz de C columnas está en el índice lineal
    i*C + j, así que su dirección es:
            dir(i, j) = base + (i*C + j) * itemsize
    De ahí que el acceso sea O(1): una fórmula de coste fijo, sin recorrer.
"""
from __future__ import annotations

import numpy as np

# ----- paleta (coherente con array_viz) -----
AZUL    = "#3b82f6"   # celda normal
NARANJA = "#f59e0b"   # celda accedida / nueva / mid de la binaria
VERDE   = "#22c55e"   # celda desplazada / encontrada / copiada
GRIS    = "#334155"   # celda descartada / fuera de rango
ROJO    = "#ef4444"   # celda borrada
BORDE   = "#1e3a8a"


# --------------------------------------------------------------------------
# helpers internos
# --------------------------------------------------------------------------
def _show(html: str):
    """Renderiza en el notebook; si no hay IPython, devuelve el HTML (para tests)."""
    try:
        from IPython.display import HTML, display
        display(HTML(html))
    except Exception:
        return html


def _check(mat) -> np.ndarray:
    mat = np.asarray(mat)
    if mat.ndim != 2:
        raise ValueError("Las visualizaciones de matrix_viz esperan un array 2-D (matriz)")
    return mat


def _celda(valor, color=AZUL, borde: str | None = None) -> str:
    b = f";border:3px solid {borde}" if borde else ""
    return (f'<div style="width:46px;height:46px;background:{color};color:#fff;'
            f'display:flex;align-items:center;justify-content:center;font-weight:bold;'
            f'border-radius:6px;font-size:14px{b}">{valor}</div>')


def _grid(mat, color_of, borde_of=None, titulo: str | None = None) -> str:
    """Dibuja la matriz como rejilla con índices de fila (izq.) y columna (arriba)."""
    R, C = mat.shape
    # cabecera de columnas
    head = ['<div style="width:24px"></div>'] + [
        f'<div style="width:46px;text-align:center;font-size:11px;color:#888">j={j}</div>'
        for j in range(C)
    ]
    filas = [f'<div style="display:flex;gap:5px;align-items:center">{"".join(head)}</div>']
    for i in range(R):
        cel = [f'<div style="width:24px;font-size:11px;color:#888">i={i}</div>']
        for j in range(C):
            borde = borde_of(i, j) if borde_of else None
            cel.append(_celda(mat[i, j], color_of(i, j), borde))
        filas.append(f'<div style="display:flex;gap:5px;align-items:center">{"".join(cel)}</div>')
    cab = (f'<div style="font-size:12px;color:#888;margin:4px 0">{titulo}</div>'
           if titulo else "")
    return (cab + '<div style="display:flex;flex-direction:column;gap:5px;'
            'font-family:sans-serif">' + "".join(filas) + '</div>')


# --------------------------------------------------------------------------
# 1. MEMORIA — la matriz 2D es un array 1D contiguo (fila-mayor)
# --------------------------------------------------------------------------
def ver_memoria(mat):
    """Muestra la rejilla y cómo se aplana a memoria contigua (row-major)."""
    mat = _check(mat)
    R, C = mat.shape
    base = mat.ctypes.data
    sz = mat.itemsize
    sfila, scol = mat.strides
    contig = mat.flags["C_CONTIGUOUS"]

    grid = _grid(mat, lambda i, j: AZUL, titulo="Vista lógica (filas × columnas)")

    # tira de memoria contigua: recorre en orden fila-mayor
    tira = []
    for i in range(R):
        for j in range(C):
            lineal = i * C + j
            dir_ = base + lineal * sz
            borde = "#f59e0b" if j == 0 else BORDE   # marca el inicio de cada fila
            tira.append(
                f'<div style="text-align:center">'
                f'<div style="width:44px;height:34px;background:{AZUL};color:#fff;'
                f'display:flex;align-items:center;justify-content:center;font-size:12px;'
                f'border:2px solid {borde}">{mat[i, j]}</div>'
                f'<div style="font-size:9px;color:#888">{lineal}</div>'
                f'<div style="font-size:9px;color:#666">({i},{j})</div></div>'
            )
    cab = (f'<div style="font-size:12px;color:#888;margin:10px 0 4px">'
           f'Memoria real (contigua, fila tras fila) · dtype={mat.dtype} · '
           f'{sz} bytes/elem · strides={mat.strides} · contiguo={contig} · '
           f'base={hex(base)}<br>índice lineal = i*{C}+j &nbsp;→&nbsp; '
           f'borde naranja = inicio de fila</div>')
    memoria = (cab + '<div style="display:flex;gap:4px;flex-wrap:wrap;'
               'font-family:sans-serif">' + "".join(tira) + '</div>')
    _show(f'<div style="font-family:sans-serif">{grid}{memoria}</div>')
    print(f"strides={mat.strides} -> avanzar 1 fila = {sfila} bytes (={C}×{sz}); "
          f"avanzar 1 columna = {scol} bytes. Todo contiguo en orden fila-mayor.")


# --------------------------------------------------------------------------
# 2. ACCESO (GET) — O(1)
# --------------------------------------------------------------------------
def ver_get(mat, i: int, j: int):
    """Resalta (i,j) y muestra la aritmética de direcciones 2D -> O(1)."""
    mat = _check(mat)
    R, C = mat.shape
    base, sz = mat.ctypes.data, mat.itemsize
    dir_ = base + i * mat.strides[0] + j * mat.strides[1]
    grid = _grid(mat, lambda a, b: NARANJA if (a, b) == (i, j) else AZUL)
    _show(f'<div style="font-family:sans-serif">{grid}</div>')
    print(f"GET [{i}][{j}] = {mat[i, j]}")
    print(f"  índice lineal = i*C + j = {i}*{C} + {j} = {i*C + j}")
    print(f"  dir = base + (i*C+j)*itemsize = {hex(base)} + {(i*C+j)*sz} = {hex(dir_)}")
    print("  -> una fórmula de coste fijo, sin recorrer nada: O(1)")


# --------------------------------------------------------------------------
# 3. ASIGNAR (SET) — O(1)
# --------------------------------------------------------------------------
def ver_set(mat, i: int, j: int, valor):
    """Antes/después: SET sobreescribe una celda en su sitio -> O(1)."""
    mat = _check(mat)
    nuevo = mat.copy(); nuevo[i, j] = valor
    antes = _grid(mat, lambda a, b: AZUL, titulo="Antes")
    despues = _grid(nuevo, lambda a, b: NARANJA if (a, b) == (i, j) else AZUL,
                    titulo=f"Después de set([{i}][{j}] = {valor})")
    _show(f'<div style="font-family:sans-serif;display:flex;gap:32px;'
          f'flex-wrap:wrap">{antes}{despues}</div>')
    print(f"SET [{i}][{j}] = {valor} -> se sobreescribe la celda en su sitio, "
          f"sin mover nada. O(1).")


# --------------------------------------------------------------------------
# 4. RECORRIDO (TRAVERSE) — O(n)  (n = nº total de elementos = R*C)
# --------------------------------------------------------------------------
def ver_traverse(mat, animar: bool = False, pausa: float = 0.25):
    """Recorre la matriz en orden fila-mayor.

    animar=True: animación por CSS (@keyframes) que resalta cada celda en
    secuencia y se repite en bucle. Un único HTML -> funciona en cualquier
    frontend y se conserva al guardar el .ipynb.
    """
    mat = _check(mat)
    R, C = mat.shape
    n = R * C
    if animar and n > 0:
        dur = round(n * pausa, 3)
        corte = round(100 / n, 4)
        anim = f"mtrav_{abs(hash((R, C, pausa))) % 100000}"
        keyframes = (f"@keyframes {anim}{{0%{{background:{NARANJA}}}"
                     f"{corte}%{{background:{NARANJA}}}"
                     f"{min(corte + 0.01, 100)}%{{background:{AZUL}}}"
                     f"100%{{background:{AZUL}}}}}")
        filas = []
        for i in range(R):
            cel = []
            for j in range(C):
                delay = round((i * C + j) * pausa, 3)   # orden fila-mayor
                cel.append(
                    f'<div style="width:46px;height:46px;background:{AZUL};color:#fff;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-weight:bold;border-radius:6px;'
                    f'animation:{anim} {dur}s linear {delay}s infinite">{mat[i, j]}</div>'
                )
            filas.append(f'<div style="display:flex;gap:5px">{"".join(cel)}</div>')
        html = (f"<style>{keyframes}</style><div style=\"font-family:sans-serif\">"
                f'<div style="font-size:12px;color:#888;margin:4px 0">'
                f'Recorrido animado (fila-mayor): la casilla resaltada avanza en bucle</div>'
                f'<div style="display:flex;flex-direction:column;gap:5px">'
                + "".join(filas) + "</div></div>")
        _show(html)
    else:
        _show(f'<div style="font-family:sans-serif">'
              f'{_grid(mat, lambda a, b: AZUL, titulo="Recorrido fila-mayor: cada celda una vez")}</div>')
    print(f"TRAVERSE -> {n} visitas ({R}×{C}) -> O(n) tiempo, O(1) espacio.")


# --------------------------------------------------------------------------
# 5. INSERCIÓN / ELIMINACIÓN de fila  (posición específica: O(n))
# --------------------------------------------------------------------------
def ver_insert_fila(mat, idx: int, fila):
    """Inserta una fila en idx: las filas siguientes se desplazan hacia abajo."""
    mat = _check(mat)
    R, C = mat.shape
    fila = np.asarray(fila).reshape(1, C)
    res = np.insert(mat, idx, fila, axis=0)
    def color(i, j):
        if i == idx:
            return NARANJA                       # fila nueva
        if i > idx:
            return VERDE                          # desplazadas hacia abajo
        return AZUL
    grid = _grid(res, color, titulo=f"insert fila en i={idx} (naranja=nueva, verde=desplazadas)")
    _show(f'<div style="font-family:sans-serif">{grid}</div>')
    print(f"Insertar fila en i={idx} -> {R - idx} filas ({(R - idx) * C} elementos) "
          f"se desplazan. O(n). (Añadir al final NO desplaza nada -> O(1) amortizado.)")


def ver_delete_fila(mat, idx: int):
    """Elimina la fila idx: las filas siguientes se desplazan hacia arriba."""
    mat = _check(mat)
    R, C = mat.shape
    antes = _grid(mat, lambda i, j: ROJO if i == idx else AZUL,
                  titulo=f"Antes (rojo = fila i={idx} que se borra)")
    res = np.delete(mat, idx, axis=0)
    despues = _grid(res, lambda i, j: VERDE if i >= idx else AZUL,
                    titulo="Después (verde = desplazadas hacia arriba)")
    _show(f'<div style="font-family:sans-serif;display:flex;gap:32px;'
          f'flex-wrap:wrap">{antes}{despues}</div>')
    print(f"Borrar fila i={idx} -> {R - idx - 1} filas ({(R - idx - 1) * C} elementos) "
          f"se desplazan. O(n). (Borrar la última NO desplaza -> O(1).)")


# --------------------------------------------------------------------------
# 6. BÚSQUEDA LINEAL — O(n)
# --------------------------------------------------------------------------
def ver_busqueda_lineal(mat, objetivo):
    """Recorre en orden fila-mayor hasta encontrar 'objetivo'. O(n)."""
    mat = _check(mat)
    R, C = mat.shape
    encontrado = None
    comparaciones = 0
    for i in range(R):
        for j in range(C):
            comparaciones += 1
            if mat[i, j] == objetivo:
                encontrado = (i, j)
                break
        if encontrado:
            break

    def color(i, j):
        lineal = i * C + j
        if encontrado and (i, j) == encontrado:
            return VERDE                          # hallado
        if encontrado and lineal < encontrado[0] * C + encontrado[1]:
            return GRIS                           # ya comprobadas (no eran)
        if not encontrado:
            return GRIS                           # todas comprobadas, no está
        return AZUL                               # aún sin visitar
    grid = _grid(mat, color, titulo=f"Búsqueda lineal de {objetivo} "
                 f"(gris=comprobadas, verde=encontrado)")
    _show(f'<div style="font-family:sans-serif">{grid}</div>')
    if encontrado:
        print(f"{objetivo} encontrado en [{encontrado[0]}][{encontrado[1]}] "
              f"tras {comparaciones} comparaciones. Peor caso O(n).")
    else:
        print(f"{objetivo} no está. Se comprobaron los {comparaciones} elementos. O(n).")


# --------------------------------------------------------------------------
# 7. BÚSQUEDA BINARIA — O(log n)  (requiere secuencia ORDENADA)
# --------------------------------------------------------------------------
def ver_busqueda_binaria(arr, objetivo):
    """Búsqueda binaria sobre un array 1D ORDENADO. Muestra cómo se parte a la mitad.

    (La binaria necesita orden; en una matriz se aplica sobre una fila
    ordenada o sobre la matriz aplanada y ordenada.)
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("La búsqueda binaria se muestra sobre un array 1D ordenado")
    n = len(arr)
    lo, hi = 0, n - 1
    pasos = []
    encontrado = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        pasos.append((lo, mid, hi))
        if arr[mid] == objetivo:
            encontrado = mid
            break
        elif arr[mid] < objetivo:
            lo = mid + 1
        else:
            hi = mid - 1

    filas_html = []
    for paso, (lo_, mid_, hi_) in enumerate(pasos):
        cel = []
        for k, v in enumerate(arr):
            if k == mid_:
                col = VERDE if (encontrado == mid_ and paso == len(pasos) - 1) else NARANJA
            elif lo_ <= k <= hi_:
                col = AZUL                        # rango activo
            else:
                col = GRIS                        # descartado
            cel.append(_celda(v, col))
        filas_html.append(
            f'<div style="display:flex;gap:5px;align-items:center;margin-bottom:4px">'
            f'<div style="width:150px;font-size:11px;color:#888">'
            f'paso {paso + 1}: lo={lo_} mid={mid_} hi={hi_}</div>{"".join(cel)}</div>'
        )
    cab = ('<div style="font-size:12px;color:#888;margin:4px 0">'
           f'Búsqueda binaria de {objetivo} (naranja=mid, azul=rango activo, '
           'gris=descartado)</div>')
    _show(f'<div style="font-family:sans-serif">{cab}{"".join(filas_html)}</div>')
    if encontrado >= 0:
        print(f"{objetivo} encontrado en índice {encontrado} tras {len(pasos)} pasos "
              f"(de {n} elementos). O(log n).")
    else:
        print(f"{objetivo} no está. {len(pasos)} pasos sobre {n} elementos. O(log n).")


__all__ = [
    "ver_memoria", "ver_get", "ver_set", "ver_traverse",
    "ver_insert_fila", "ver_delete_fila",
    "ver_busqueda_lineal", "ver_busqueda_binaria",
]
