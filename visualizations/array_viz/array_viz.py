"""
array_viz — visualizaciones de Arrays para el notebook de la lección.

Todas las funciones reciben un numpy.ndarray (1-D) como entrada y dibujan
el resultado en la salida de la celda con IPython.display.HTML.

Cubre todas las operaciones de la lección:
    memoria · init · get · set · traverse · copy · insert · delete · duplicacion

Nota de fidelidad:
    - numpy guarda los valores en CRUDO y CONTIGUOS (como la zona de datos de
      un array de C) y es de TAMAÑO FIJO -> es el mejor análogo de un array
      ESTÁTICO. Por eso get/set/traverse/copy/memoria usan numpy de verdad.
    - insert/delete en numpy crean un array NUEVO (copia O(N)); lo reflejamos.
    - La capacidad reservada (duplicación) es propia de arrays DINÁMICOS y
      numpy no la expone, así que `ver_duplicacion` la modela explícitamente.
"""
from __future__ import annotations

import numpy as np

# ----- paleta -----
AZUL   = "#3b82f6"   # elemento usado
NARANJA = "#f59e0b"  # elemento accedido / nuevo
VERDE  = "#22c55e"   # elemento desplazado / copiado
GRIS   = "#1c2138"   # celda reservada libre
ROJO   = "#ef4444"   # elemento borrado
BORDE  = "#1e3a8a"


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


def _caja(valor, idx, color=AZUL, nota: str | None = None) -> str:
    """Una celda 'de alto nivel' (un elemento), con su índice debajo."""
    sub = nota if nota is not None else f"[{idx}]"
    return (
        f'<div style="text-align:center">'
        f'<div style="width:48px;height:48px;background:{color};color:#fff;'
        f'display:flex;align-items:center;justify-content:center;font-weight:bold;'
        f'border-radius:8px">{valor}</div>'
        f'<div style="font-size:11px;color:#888;margin-top:3px">{sub}</div></div>'
    )


def _fila(cajas: list[str], titulo: str | None = None) -> str:
    cab = (f'<div style="font-size:12px;color:#888;margin:4px 0">{titulo}</div>'
           if titulo else "")
    return (cab + '<div style="display:flex;gap:6px;flex-wrap:wrap;'
            'align-items:flex-end">' + "".join(cajas) + '</div>')


def _check(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("Las visualizaciones esperan un array 1-D")
    return arr


# --------------------------------------------------------------------------
# 1. MEMORIA — celdas contiguas, bytes reales
# --------------------------------------------------------------------------
def ver_memoria(arr):
    """Dibuja el array byte a byte usando dirección, itemsize y strides REALES."""
    arr = _check(arr)
    base = arr.ctypes.data            # dirección real del buffer
    sz = arr.itemsize                 # bytes por elemento (4=int32, 8=int64/float64)
    paso = arr.strides[0]             # salto en bytes entre elementos -> contigüidad
    contig = arr.flags["C_CONTIGUOUS"]

    cols = []
    for i, v in enumerate(arr):
        ini = base + i * paso
        celdas = "".join(
            f'<div style="width:30px;height:32px;background:{AZUL};color:#fff;'
            f'display:flex;align-items:center;justify-content:center;font-size:9px;'
            f'border:1px solid {BORDE}">{hex(ini + b)[-3:]}</div>'
            for b in range(sz)
        )
        cols.append(
            f'<div style="text-align:center"><div style="display:flex">{celdas}</div>'
            f'<div style="font-size:12px;color:#2563eb;margin-top:3px">[{i}] = {v}</div>'
            f'<div style="font-size:10px;color:#888">{hex(ini)}</div></div>'
        )
    cab = (f'<div style="font-size:12px;color:#888;margin-bottom:6px">'
           f'dtype={arr.dtype} · {sz} bytes/elem · strides={arr.strides} '
           f'· contiguo={contig} · base={hex(base)}</div>')
    return _show(f'<div style="font-family:sans-serif">{cab}'
                 f'<div style="display:flex;gap:6px;flex-wrap:wrap">{"".join(cols)}</div></div>')


# --------------------------------------------------------------------------
# 2. INIT — reservar e inicializar N celdas  (O(N))
# --------------------------------------------------------------------------
def ver_init(arr):
    """Muestra un array recién inicializado: hay que TOCAR las N celdas -> O(N)."""
    arr = _check(arr)
    cajas = [_caja(v, i, AZUL) for i, v in enumerate(arr)]
    out = _fila(cajas, f"Init de {len(arr)} elementos (dtype={arr.dtype})")
    _show(f'<div style="font-family:sans-serif">{out}</div>')
    print(f"Init -> reservar + escribir {len(arr)} celdas = O(N) tiempo y espacio.")


# --------------------------------------------------------------------------
# 3. GET — acceso por índice  (O(1))
# --------------------------------------------------------------------------
def ver_get(arr, idx: int):
    """Resalta el elemento accedido y muestra la aritmética de direcciones O(1)."""
    arr = _check(arr)
    base, sz = arr.ctypes.data, arr.itemsize
    cajas = [_caja(v, i, NARANJA if i == idx else AZUL) for i, v in enumerate(arr)]
    _show(f'<div style="font-family:sans-serif">{_fila(cajas)}</div>')
    print(f"GET [{idx}] = {arr[idx]} -> dir = base + {idx}*{sz} = "
          f"{hex(base)} + {idx*sz} = {hex(base + idx*sz)}   (acceso directo, O(1))")


# --------------------------------------------------------------------------
# 4. SET — asignar un valor  (O(1), NO mueve memoria)
# --------------------------------------------------------------------------
def ver_set(arr, idx: int, valor):
    """Muestra antes/después: SET sobreescribe una celda en su sitio -> O(1)."""
    arr = _check(arr)
    antes = [_caja(v, i, AZUL) for i, v in enumerate(arr)]
    nuevo = arr.copy(); nuevo[idx] = valor
    despues = [_caja(v, i, NARANJA if i == idx else AZUL) for i, v in enumerate(nuevo)]
    out = (_fila(antes, "Antes") +
           _fila(despues, f"Después de set([{idx}] = {valor})"))
    _show(f'<div style="font-family:sans-serif">{out}</div>')
    print(f"SET [{idx}] = {valor} -> se sobreescribe la celda en su sitio, "
          f"sin desplazar nada. O(1).")


# --------------------------------------------------------------------------
# 5. TRAVERSE — recorrer  (O(N) tiempo, O(1) espacio)
# --------------------------------------------------------------------------
def ver_traverse(arr, animar: bool = False, pausa: float = 0.35):
    """Recorre el array.

    animar=True usa una animación por CSS (@keyframes): la casilla resaltada
    avanza en bucle. Es un único HTML, así que funciona en cualquier frontend
    (Jupyter Lab/Notebook, VS Code) y se conserva al guardar el .ipynb, a
    diferencia de las animaciones por bucle+time.sleep, que bloquean el kernel.
    """
    arr = _check(arr)
    n = len(arr)
    if animar and n > 0:
        dur = round(n * pausa, 3)                 # duración de un ciclo completo
        corte = round(100 / n, 4)                 # % del ciclo que cada casilla está resaltada
        anim = f"trav_{abs(hash((n, pausa))) % 100000}"
        keyframes = (
            f"@keyframes {anim}{{"
            f"0%{{background:{NARANJA}}}"
            f"{corte}%{{background:{NARANJA}}}"
            f"{min(corte + 0.01, 100)}%{{background:{AZUL}}}"
            f"100%{{background:{AZUL}}}}}"
        )
        cajas = []
        for i, v in enumerate(arr):
            delay = round(i * pausa, 3)           # cada casilla arranca un paso después
            cajas.append(
                f'<div style="text-align:center">'
                f'<div style="width:48px;height:48px;background:{AZUL};color:#fff;'
                f'display:flex;align-items:center;justify-content:center;font-weight:bold;'
                f'border-radius:8px;animation:{anim} {dur}s linear {delay}s infinite">{v}</div>'
                f'<div style="font-size:11px;color:#888;margin-top:3px">[{i}]</div></div>'
            )
        html = (
            f"<style>{keyframes}</style>"
            f'<div style="font-family:sans-serif">'
            f'<div style="font-size:12px;color:#888;margin:4px 0">'
            f'Recorrido animado: la casilla resaltada avanza y se repite en bucle</div>'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:flex-end">'
            + "".join(cajas) + "</div></div>"
        )
        _show(html)
    else:
        # estático: numeramos el orden de visita 0..N-1
        cajas = [_caja(v, i, AZUL, nota=f"paso {i}") for i, v in enumerate(arr)]
        _show(f'<div style="font-family:sans-serif">'
              f'{_fila(cajas, "Recorrido: se visita cada elemento una vez")}</div>')
    print(f"TRAVERSE -> {n} visitas, 1 contador -> O(N) tiempo, O(1) espacio.")


# --------------------------------------------------------------------------
# 6. COPY — copiar  (O(N) tiempo y espacio: buffer NUEVO)
# --------------------------------------------------------------------------
def ver_copy(arr):
    """Muestra original y copia en DIRECCIONES distintas -> nuevo buffer O(N)."""
    arr = _check(arr)
    copia = arr.copy()
    orig = [_caja(v, i, AZUL) for i, v in enumerate(arr)]
    cop = [_caja(v, i, VERDE) for i, v in enumerate(copia)]
    out = (_fila(orig, f"Original  @ {hex(arr.ctypes.data)}") +
           _fila(cop, f"Copia     @ {hex(copia.ctypes.data)}  (otro bloque de memoria)"))
    _show(f'<div style="font-family:sans-serif">{out}</div>')
    print(f"COPY -> se reserva un buffer nuevo y se copian {len(arr)} elementos. "
          f"O(N) tiempo y espacio. ¿Misma dirección? "
          f"{arr.ctypes.data == copia.ctypes.data}")


# --------------------------------------------------------------------------
# 7. INSERT — insertar en índice  (O(N): desplazar)
# --------------------------------------------------------------------------
def ver_insert(arr, idx: int, valor):
    """np.insert crea un array nuevo; marcamos el nuevo y los desplazados."""
    arr = _check(arr)
    n = len(arr)
    res = np.insert(arr, idx, valor)
    estado = {idx: NARANJA}                       # el nuevo
    for k in range(idx, n):
        estado[k + 1] = VERDE                     # desplazados a la derecha
    cajas = [_caja(v, i, estado.get(i, AZUL)) for i, v in enumerate(res)]
    _show(f'<div style="font-family:sans-serif">{_fila(cajas)}</div>')
    print(f"insert({idx}, {valor}) -> {n - idx} desplazamientos a la derecha (verde). "
          f"Naranja = nuevo. O(N).")


# --------------------------------------------------------------------------
# 8. DELETE — borrar en índice  (O(N): desplazar)
# --------------------------------------------------------------------------
def ver_delete(arr, idx: int):
    """Muestra el elemento borrado (rojo) y el desplazamiento a la izquierda."""
    arr = _check(arr)
    n = len(arr)
    antes = [_caja(v, i, ROJO if i == idx else AZUL) for i, v in enumerate(arr)]
    res = np.delete(arr, idx)
    estado = {i: VERDE for i in range(idx, len(res))}   # los que se mueven
    despues = [_caja(v, i, estado.get(i, AZUL)) for i, v in enumerate(res)]
    out = (_fila(antes, f"Antes (rojo = se borra [{idx}])") +
           _fila(despues, "Después (verde = desplazado a la izquierda)"))
    _show(f'<div style="font-family:sans-serif">{out}</div>')
    print(f"delete({idx}) -> {n - idx - 1} desplazamientos a la izquierda para "
          f"tapar el hueco. O(N).")


# --------------------------------------------------------------------------
# 9. DUPLICACIÓN — capacidad de un array DINÁMICO
# --------------------------------------------------------------------------
def ver_duplicacion(arr, capacidad: int):
    """Modela un array dinámico: `arr` son los usados, `capacidad` los reservados.

    numpy NO expone capacidad (es estático), así que la pasamos a mano para
    ilustrar por qué `append` es O(1) amortizado: se reserva de más.
    """
    arr = _check(arr)
    usados = len(arr)
    if capacidad < usados:
        raise ValueError("capacidad debe ser >= número de elementos usados")
    celdas = []
    for i in range(capacidad):
        if i < usados:
            bg, txt = AZUL, str(arr[i])
        else:
            bg, txt = GRIS, ""        # hueco reservado por la duplicación
        celdas.append(
            f'<div style="width:42px;height:42px;background:{bg};color:#fff;'
            f'border:1px dashed #555;display:flex;align-items:center;'
            f'justify-content:center;font-size:12px">{txt}</div>'
        )
    out = ('<div style="display:flex;gap:4px;flex-wrap:wrap">' +
           "".join(celdas) + '</div>')
    _show(f'<div style="font-family:sans-serif">{out}</div>')
    print(f"len={usados}  capacidad={capacidad}  ocupación={round(usados/capacidad*100)}%  "
          f"-> los huecos reservados hacen que append sea O(1) amortizado.")


__all__ = [
    "ver_memoria", "ver_init", "ver_get", "ver_set", "ver_traverse",
    "ver_copy", "ver_insert", "ver_delete", "ver_duplicacion",
]
