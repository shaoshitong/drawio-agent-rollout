"""Decode bounded draw.io page models for the native comparison path."""

import base64
import urllib.parse
from xml.etree import ElementTree as ET
import zlib

LIMIT = 64 * 1024**2


def models(xml):
    blob = xml.encode() if isinstance(xml, str) else xml
    if len(blob) > LIMIT or b"<!DOCTYPE" in blob.upper() or b"<!ENTITY" in blob.upper():
        raise ValueError("XML_size_or_DTD_not_supported")
    root = ET.fromstring(blob)
    if root.tag == "mxGraphModel":
        return root, [root]
    if root.tag != "mxfile":
        raise ValueError("not_drawio_XML")
    result = []
    for page in root.findall("diagram"):
        model = page.find("mxGraphModel")
        if model is None:
            text = (page.text or "").strip()
            if text.startswith("<"):
                decoded = text.encode()
            else:
                compressed = base64.b64decode("".join(text.split()), validate=True)
                decoder = zlib.decompressobj(-15)
                encoded = decoder.decompress(compressed, LIMIT + 1)
                if len(encoded) > LIMIT or decoder.unconsumed_tail or not decoder.eof or decoder.unused_data:
                    raise ValueError("invalid_or_unbounded_compressed_model")
                decoded = urllib.parse.unquote(encoded.decode(), errors="strict").encode()
            if len(decoded) > LIMIT or b"<!DOCTYPE" in decoded.upper() or b"<!ENTITY" in decoded.upper():
                raise ValueError("decoded_model_size_or_DTD_not_supported")
            model = ET.fromstring(decoded)
        if model.tag != "mxGraphModel":
            raise ValueError("decoded_page_not_graph_model")
        result.append(model)
    if not result:
        raise ValueError("no_graph_model")
    return root, result
