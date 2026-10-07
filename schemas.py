"""
Esquemas de datos por tipo de documento.

Cada dato extraído es un `Campo` con dos partes:
  - valor:     el dato tal como aparece en el documento ("" si no aparece)
  - evidencia: documento y texto literal de donde salió ("" si no hay)

Para agregar un tipo de documento nuevo:
  1. Crea una clase como `PoderSimple` con sus campos.
  2. Defínele NOMBRE, PLANTILLA, OPCIONALES, ETIQUETAS, SECCIONES y ROLES_ARCHIVO.
  3. Regístrala en TIPOS_DOCUMENTO al final del archivo.
"""
from typing import ClassVar, Iterator, TypeVar

from pydantic import BaseModel, Field

M = TypeVar("M", bound=BaseModel)

# Evidencia que se asigna a lo que la persona usuaria escribió o corrigió en la revisión
EVIDENCIA_MANUAL = "Capturado o corregido por la persona usuaria"


class Campo(BaseModel):
    # Ambos son texto obligatorio, sin valores nulos: la API limita los parámetros con
    # tipos "unión" (como str | null) a 16 por solicitud, y este esquema tiene muchos campos.
    # Cuando el dato no aparece, el modelo debe devolver una cadena vacía.
    valor: str = Field(
        description='Dato tal como aparece en el documento. Cadena vacía "" si no aparece claramente.'
    )
    evidencia: str = Field(
        description='Nombre del documento y texto literal breve de donde se sacó el dato. Cadena vacía "" si valor está vacío.'
    )


def recorrer_campos(modelo: BaseModel, prefijo: str = "") -> Iterator[tuple[str, "Campo"]]:
    """
    Recorre un modelo (incluidos los modelos anidados) y entrega cada Campo con su ruta,
    por ejemplo ("otorgante.curp", Campo(...)). Las rutas son las mismas que usan
    OPCIONALES, ETIQUETAS y los nombres de los campos del formulario de revisión.
    """
    for nombre in type(modelo).model_fields:
        valor = getattr(modelo, nombre)
        ruta = f"{prefijo}{nombre}"
        if isinstance(valor, Campo):
            yield ruta, valor
        elif isinstance(valor, BaseModel):
            yield from recorrer_campos(valor, f"{ruta}.")


def aplicar_valores(datos: M, valores: dict[str, str]) -> M:
    """
    Devuelve una COPIA de `datos` con los valores editados en el formulario de revisión.

    - `valores` usa como clave la ruta del campo (por ejemplo "otorgante.curp").
    - Las rutas que no vienen en `valores` conservan su dato original; las desconocidas se ignoran.
    - Si la persona cambió un valor (o llenó uno vacío), la evidencia pasa a ser EVIDENCIA_MANUAL,
      para que un dato escrito por ella misma no se marque como "sin evidencia".
    - Si lo dejó vacío, la evidencia también queda vacía.
    - Si no lo cambió, se conserva la evidencia original.
    """
    nuevo = datos.model_copy(deep=True)
    for ruta, campo in recorrer_campos(nuevo):
        if ruta not in valores:
            continue
        bruto = (valores[ruta] or "").strip()
        if bruto == (campo.valor or "").strip():
            continue  # sin cambios: se conserva el dato y su evidencia originales
        editado = _normalizar(ruta, bruto)
        campo.valor = editado
        campo.evidencia = EVIDENCIA_MANUAL if editado else ""
    return nuevo


def _normalizar(ruta: str, texto: str) -> str:
    """CURP y RFC se escriben en mayúsculas y sin espacios, aunque la persona los teclee distinto."""
    if ruta.rsplit(".", 1)[-1] in ("curp", "rfc"):
        return texto.replace(" ", "").upper()
    return texto


def _estructura_vacia(clase: type[BaseModel]) -> dict:
    estructura: dict = {}
    for nombre, info in clase.model_fields.items():
        anotacion = info.annotation
        if anotacion is Campo:
            estructura[nombre] = {"valor": "", "evidencia": ""}
        elif isinstance(anotacion, type) and issubclass(anotacion, BaseModel):
            estructura[nombre] = _estructura_vacia(anotacion)
        else:
            raise TypeError(f"Tipo no soportado en {clase.__name__}.{nombre}: se esperaba Campo o un modelo anidado")
    return estructura


def crear_vacio(tipo: type[M]) -> M:
    """
    Instancia de `tipo` con TODOS los campos vacíos. Sirve para el modo manual: la persona
    captura los datos desde cero en la pantalla de revisión, sin subir documentos.
    """
    return tipo.model_validate(_estructura_vacia(tipo))


class Persona(BaseModel):
    nombre_completo: Campo = Field(description="Nombre completo, tal como aparece en el documento")
    curp: Campo = Field(description="CURP (18 caracteres)")
    rfc: Campo = Field(description="RFC, solo si aparece escrito en algún documento")
    domicilio: Campo = Field(description="Domicilio completo (calle, número, colonia, municipio, estado, C.P.)")


class PoderSimple(BaseModel):
    """Tipo de documento de EJEMPLO. Se reemplazará con los formatos reales de la notaría."""

    NOMBRE: ClassVar[str] = "Poder"
    PLANTILLA: ClassVar[str] = "poder_simple.docx"

    # Rutas de campos que pueden quedar vacíos sin marcarse como "faltante"
    OPCIONALES: ClassVar[set[str]] = {"otorgante.rfc", "apoderado.rfc"}

    # Texto legible para los mensajes que ve la persona usuaria ("Falta ...")
    ETIQUETAS: ClassVar[dict[str, str]] = {
        "otorgante.nombre_completo": "el nombre del otorgante",
        "otorgante.curp": "la CURP del otorgante",
        "otorgante.rfc": "el RFC del otorgante",
        "otorgante.domicilio": "el domicilio del otorgante",
        "apoderado.nombre_completo": "el nombre del apoderado",
        "apoderado.curp": "la CURP del apoderado",
        "apoderado.rfc": "el RFC del apoderado",
        "apoderado.domicilio": "el domicilio del apoderado",
        "facultades": "el detalle de las facultades que se otorgan",
        "lugar_y_fecha": "el lugar y la fecha del otorgamiento",
    }

    # Títulos de sección en la pantalla de revisión (prefijo de la ruta -> título).
    # Los campos de primer nivel (sin prefijo) se agrupan en "Datos generales".
    SECCIONES: ClassVar[dict[str, str]] = {
        "otorgante": "Otorgante (quien da el poder)",
        "apoderado": "Apoderado (quien recibe el poder)",
    }

    # Roles que la persona usuaria elige para cada archivo que sube. La etiqueta elegida
    # se envía al modelo para que sepa de quién es cada documento.
    ROLES_ARCHIVO: ClassVar[list[str]] = [
        "Identificación del otorgante",
        "Identificación del apoderado",
        "Otro documento",
    ]

    otorgante: Persona = Field(description="Persona que otorga el poder")
    apoderado: Persona = Field(description="Persona que recibe el poder")
    facultades: Campo = Field(description="Facultades o tipo de poder que se otorga, solo si se indica explícitamente")
    lugar_y_fecha: Campo = Field(description="Lugar y fecha del otorgamiento, solo si se indican explícitamente")


# Nombre interno -> clase. La interfaz usa estas claves para el paso 1.
TIPOS_DOCUMENTO: dict[str, type[BaseModel]] = {
    "poder": PoderSimple,
}
