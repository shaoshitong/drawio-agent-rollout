"""Score model drawio against the teacher final drawio using drawio.compare evidence."""
import json, math, re, sys
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_color(style):
    m = re.search(r'fillColor=(#[0-9a-fA-F]{6}|none)', style or '')
    if not m or m.group(1) == 'none':
        return None
    h = m.group(1)[1:]
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def srgb_to_lab(rgb):
    def f(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (f(v) for v in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b)
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    def g(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = g(x), g(y), g(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))

def de2000(lab1, lab2):
    L1, a1, b1 = lab1; L2, a2, b2 = lab2
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7))) if Cm else 0.5
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = 0 if C1p * C2p == 0 else (h2p - h1p if abs(h2p - h1p) <= 180 else (h2p - h1p - 360 if h2p - h1p > 180 else h2p - h1p + 360))
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    hpm = h1p + h2p
    if C1p * C2p == 0:
        hpm = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hpm = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hpm = (h1p + h2p + 360) / 2
    else:
        hpm = (h1p + h2p - 360) / 2
    T = (1 - 0.17 * math.cos(math.radians(hpm - 30)) + 0.24 * math.cos(math.radians(2 * hpm))
         + 0.32 * math.cos(math.radians(3 * hpm + 6)) - 0.20 * math.cos(math.radians(4 * hpm - 63)))
    dTheta = 30 * math.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(Cpm ** 7 / (Cpm ** 7 + 25 ** 7)) if Cpm else 0
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / math.sqrt(20 + (Lpm - 50) ** 2)
    Sc = 1 + 0.045 * Cpm
    Sh = 1 + 0.015 * Cpm * T
    Rt = -math.sin(math.radians(2 * dTheta)) * Rc
    return math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))

def counts_from(diff_path):
    d = json.loads(Path(diff_path).read_text())
    ms = d['pages'][0]['matches']
    good = [m for m in ms if m.get('status') == 'confirmed' and m.get('score', 0) >= 0.9]
    return (sum(1 for m in good if m['a']['kind'] == 'shape'), sum(1 for m in good if m['a']['kind'] == 'edge'))

def score(compare_dir, teacher_xml, model_xml, teacher_self, model_self):
    d = json.loads((Path(compare_dir) / 'diff.json').read_text())
    page = d['pages'][0]
    matches = page['matches']
    frame = page.get('frame') or {}
    diag = math.hypot(frame.get('width', 0), frame.get('height', 0)) or 1.0
    good = [m for m in matches if m.get('status') == 'confirmed' and m.get('score', 0) >= 0.9]
    shapes = [m for m in good if m['a']['kind'] == 'shape']
    edges = [m for m in good if m['a']['kind'] == 'edge']
    # totals from XML
    av, ae = counts_from(teacher_self); bv, be = counts_from(model_self)
    def f1(m, a, b):
        p = m / a if a else 0.0; r = m / b if b else 0.0
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)
    # position error over matched shapes
    pos_err = []
    color_pairs = []
    colors_a, colors_b = {}, {}
    for path, store in ((teacher_xml, colors_a), (model_xml, colors_b)):
        root = ET.parse(path).getroot()
        for c in root.iter('mxCell'):
            if c.get('vertex') == '1':
                store[c.get('id')] = parse_color(c.get('style'))
    for m in shapes:
        ax, ay, aw, ah = m['a']['bbox_page_xywh']; bx, by, bw, bh = m['b']['bbox_page_xywh']
        pos_err.append(math.hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2)) / diag)
        ca = next((colors_a.get(i) for i in m['a']['cell_ids'] if colors_a.get(i)), None)
        cb = next((colors_b.get(i) for i in m['b']['cell_ids'] if colors_b.get(i)), None)
        if ca and cb:
            color_pairs.append(de2000(srgb_to_lab(ca), srgb_to_lab(cb)))
    pos_ok = sum(1 for e in pos_err if e <= 0.05) / len(pos_err) if pos_err else 0.0
    color_ok = sum(1 for e in color_pairs if e <= 10) / len(color_pairs) if color_pairs else 0.0
    return {"structure_f1": round(f1(len(shapes), av, bv), 4), "edge_f1": round(f1(len(edges), ae, be), 4),
            "matched_shapes": len(shapes), "teacher_shapes": av, "model_shapes": bv,
            "position_within_5pct": round(pos_ok, 4), "color_within_de10": round(color_ok, 4),
            "colors_compared": len(color_pairs)}

if __name__ == '__main__':
    print(json.dumps(score(*sys.argv[1:6]), ensure_ascii=False))
