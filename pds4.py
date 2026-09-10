"""
Reader for PDS4 products, the format ISRO distributes Chandrayaan-2 data in.

Products from https://chmapbrowse.issdc.gov.in/ arrive as a detached pair: an
XML label describing the array, and a raw binary file holding it. There is no
header in the binary itself, so the label is the only way to know the
dimensions, element type and byte order.

    from pds4 import read_product, to_display_image

    product = read_product("ch2_ohr_ncp_20200827.xml")
    image = to_display_image(product)          # 8-bit, ready for the pipeline

Handles Array_2D_Image (OHRC, TMC-2) and Array_3D_Spectrum (IIRS), including
band selection, scaling factors and missing-data constants.

Written against the PDS4 Information Model 1.x specification. Chandrayaan-2
products vary in which optional elements they carry, so anything not required
by the spec is treated as optional here.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

# PDS4 data types map onto numpy dtype strings. LSB/MSB in the PDS name gives
# the byte order; numpy needs that as an explicit "<" or ">" prefix, because a
# native-order read would silently produce garbage on the wrong architecture.
DATA_TYPES = {
    "IEEE754LSBSingle": "<f4",
    "IEEE754MSBSingle": ">f4",
    "IEEE754LSBDouble": "<f8",
    "IEEE754MSBDouble": ">f8",
    "SignedLSB2": "<i2",
    "SignedMSB2": ">i2",
    "SignedLSB4": "<i4",
    "SignedMSB4": ">i4",
    "SignedLSB8": "<i8",
    "SignedMSB8": ">i8",
    "UnsignedLSB2": "<u2",
    "UnsignedMSB2": ">u2",
    "UnsignedLSB4": "<u4",
    "UnsignedMSB4": ">u4",
    "UnsignedByte": "u1",
    "SignedByte": "i1",
}

ARRAY_TAGS = ("Array_3D_Spectrum", "Array_3D_Image", "Array_2D_Image", "Array_2D", "Array")


class PDS4Error(Exception):
    """Raised when a label cannot be parsed or does not match its data file."""


def _local(tag):
    """Strip the XML namespace, which varies by product and discipline dictionary."""
    return tag.rsplit("}", 1)[-1]


def _find(element, name):
    for child in element.iter():
        if _local(child.tag) == name:
            return child
    return None


def _find_all(element, name):
    return [child for child in element.iter() if _local(child.tag) == name]


def _text(element, name, default=None):
    found = _find(element, name)
    if found is None or found.text is None:
        return default
    return found.text.strip()


def _parse_axes(array_element):
    """
    Axis order follows sequence_number, not document order.

    The spec allows the Axis_Array elements to appear in any order, with
    sequence_number giving the actual memory layout. Trusting document order
    reads a transposed image on any product that lists them differently.
    """
    axes = []
    for axis in _find_all(array_element, "Axis_Array"):
        name = _text(axis, "axis_name", "")
        elements = _text(axis, "elements")
        sequence = _text(axis, "sequence_number", "0")
        if elements is None:
            raise PDS4Error(f"axis {name!r} has no element count")
        axes.append((int(sequence), name, int(elements)))

    if not axes:
        raise PDS4Error("array declares no axes")

    axes.sort(key=lambda item: item[0])
    return [(name, count) for _, name, count in axes]


def read_product(label_path, band=None):
    """
    Read a PDS4 product and return its array plus the metadata needed to use it.

    `band` selects a plane from a 3D spectral cube (IIRS). The default takes the
    middle band, which for IIRS is a reasonable compromise: the shortest bands
    are noisy and the longest carry thermal emission rather than reflectance.
    """
    if not os.path.exists(label_path):
        raise PDS4Error(f"label not found: {label_path}")

    try:
        root = ET.parse(label_path).getroot()
    except ET.ParseError as exc:
        raise PDS4Error(f"could not parse label: {exc}") from exc

    file_area = _find(root, "File_Area_Observational")
    if file_area is None:
        raise PDS4Error("label has no File_Area_Observational")

    file_name = _text(file_area, "file_name")
    if not file_name:
        raise PDS4Error("label does not name a data file")

    data_path = os.path.join(os.path.dirname(os.path.abspath(label_path)), file_name)
    if not os.path.exists(data_path):
        raise PDS4Error(f"data file named by the label is missing: {file_name}")

    array = None
    for tag in ARRAY_TAGS:
        array = _find(file_area, tag)
        if array is not None:
            break
    if array is None:
        raise PDS4Error("label contains no recognised Array element")

    axes = _parse_axes(array)
    shape = [count for _, count in axes]

    pds_type = _text(array, "data_type")
    dtype = DATA_TYPES.get(pds_type)
    if dtype is None:
        raise PDS4Error(f"unsupported data type {pds_type!r}")

    offset_element = _find(array, "offset")
    offset = int(offset_element.text) if offset_element is not None else 0

    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    available = os.path.getsize(data_path) - offset
    if available < expected:
        raise PDS4Error(
            f"{file_name} holds {available} bytes after the offset but the label "
            f"describes {expected} for a {'x'.join(map(str, shape))} array"
        )

    data = np.fromfile(data_path, dtype=dtype, count=int(np.prod(shape)), offset=offset)
    data = data.reshape(shape)

    axis_names = [name.lower() for name, _ in axes]
    band_index = None

    if data.ndim == 3:
        # Find which axis carries the spectral bands; the remaining two are the
        # image. IIRS labels call it "Band", but the order is not guaranteed.
        band_axis = next(
            (i for i, name in enumerate(axis_names) if "band" in name or "wavelength" in name),
            0,
        )
        band_count = shape[band_axis]
        band_index = band_count // 2 if band is None else int(band)
        if not 0 <= band_index < band_count:
            raise PDS4Error(f"band {band_index} out of range for {band_count} bands")
        data = np.take(data, band_index, axis=band_axis)
    elif data.ndim != 2:
        raise PDS4Error(f"expected a 2D or 3D array, got {data.ndim} dimensions")

    data = data.astype(np.float64)

    # Missing-data constants must be masked before any scaling, or they become
    # ordinary-looking values and drag the contrast stretch with them.
    invalid = np.zeros(data.shape, dtype=bool)
    special = _find(array, "Special_Constants")
    if special is not None:
        for name in ("missing_constant", "invalid_constant", "saturated_constant",
                     "high_instrument_saturation", "low_instrument_saturation"):
            value = _text(special, name)
            if value is not None:
                invalid |= data == float(value)

    scaling = float(_text(array, "scaling_factor", "1") or 1)
    value_offset = float(_text(array, "value_offset", "0") or 0)
    if scaling != 1.0 or value_offset != 0.0:
        data = data * scaling + value_offset

    data[invalid] = np.nan

    return {
        "data": data,
        "valid_mask": ~invalid,
        "shape": data.shape,
        "dtype": pds_type,
        "axes": axes,
        "band": band_index,
        "data_file": file_name,
        "label": {
            "logical_identifier": _text(root, "logical_identifier"),
            "title": _text(root, "title"),
            "instrument": _text(root, "instrument_name") or _text(root, "Observing_System_Component"),
            "start_time": _text(root, "start_date_time"),
        },
    }


def to_display_image(product, low=2.0, high=98.0):
    """
    Convert a product's array to the 8-bit greyscale the pipeline expects.

    Stretched between percentiles rather than min/max: instrument data routinely
    contains a handful of saturated or dead pixels, and a min/max stretch lets
    those flatten everything else to a narrow band of grey.
    """
    data = np.asarray(product["data"], dtype=np.float64)
    finite = data[np.isfinite(data)]

    if finite.size == 0:
        raise PDS4Error("array contains no valid samples")

    lo, hi = np.percentile(finite, (low, high))
    if hi - lo < 1e-12:
        return np.zeros(data.shape, dtype=np.uint8)

    stretched = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    stretched[~np.isfinite(data)] = 0.0
    return (stretched * 255).astype(np.uint8)


def describe(product):
    """One-line summary for logs and the CLI."""
    label = product["label"]
    name = label.get("logical_identifier") or product["data_file"]
    rows, cols = product["shape"]
    band = f", band {product['band']}" if product["band"] is not None else ""
    return f"{name}: {cols}x{rows} {product['dtype']}{band}"
