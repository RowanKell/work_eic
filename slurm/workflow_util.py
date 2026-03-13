"""
workflow_util.py — Shared utilities for MOBO simulation workflow scripts.

Provides XML geometry readers and memory limit logic for both the basic
(num_layers × steel_ratio) and preshower (preshower_steel_value ×
division_layer_number) MOBO parameter configurations.

Used by:
  submit_workflow.py
  test_memory_limits.py
  test_memory_limits_preshower.py
"""

import xml.etree.ElementTree as ET
import os


# ── XML Readers ────────────────────────────────────────────────────────────
#
# Each reader searches the compact file directly first, then follows any
# <include ref="...klmws..."> links to the klmws sub-file.

def _read_xml_constant(compact_file, constant_name):
    """Generic helper: return the 'value' attribute of a <constant name=X> element."""
    def _find(filepath):
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            for const in root.iter('constant'):
                if const.get('name') == constant_name:
                    return const.get('value', None)
        except Exception:
            pass
        return None

    result = _find(compact_file)
    if result is not None:
        return result

    try:
        tree = ET.parse(compact_file)
        root = tree.getroot()
        parent_dir = os.path.dirname(compact_file)
        for inc in root.iter('include'):
            ref = inc.get('ref', '')
            if 'klmws' in ref:
                resolved = ref.replace('${DETECTOR_PATH}', parent_dir)
                result = _find(resolved)
                if result is not None:
                    return result
    except Exception:
        pass
    return None


def get_scint_thickness_mm(compact_file):
    """Read HcalScintillatorThickness. Returns float (mm) or None."""
    val = _read_xml_constant(compact_file, 'HcalScintillatorThickness')
    if val is None:
        return None
    try:
        return float(val.replace('*mm', ''))
    except ValueError:
        return None


def get_num_layers_from_xml(compact_file):
    """Read HcalScintillatorNbLayers. Returns int or None."""
    val = _read_xml_constant(compact_file, 'HcalScintillatorNbLayers')
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def get_division_layer_from_xml(compact_file):
    """Read division_layer_number. Returns int or None."""
    val = _read_xml_constant(compact_file, 'division_layer_number')
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def get_preshower_steel_from_xml(compact_file):
    """Read preshower_steel_value. Returns float (mm) or None."""
    val = _read_xml_constant(compact_file, 'preshower_steel_value')
    if val is None:
        return None
    try:
        return float(val.replace('*mm', ''))
    except ValueError:
        return None


def get_steel_thickness_mm(compact_file):
    """Read HcalSteelThickness. Returns float (mm) or None."""
    val = _read_xml_constant(compact_file, 'HcalSteelThickness')
    if val is None:
        return None
    try:
        return float(val.replace('*mm', ''))
    except ValueError:
        return None


# ── Memory Limit Logic ─────────────────────────────────────────────────────

def get_mem_limit(compact_file, particle_name, geo_config="basic"):
    """
    Return the SLURM --mem string for a given (compact_file, particle, geo_config).

    geo_config choices:
      "basic"     — original parameter space: num_layers × steel_ratio
                    Limits driven by HcalScintillatorThickness and HcalScintillatorNbLayers.
                    Validated by test_memory_limits.py.

      "preshower" — preshower parameter space: preshower_steel_value × division_layer_number
                    Limits driven by division_layer_number (cell count) and
                    preshower_steel_value (absorption; thin → more downstream hits).
                    NOTE: initial tiers are conservative placeholders — update after
                    running test_memory_limits_preshower.py.
    """
    if geo_config == "preshower":
        return _get_mem_limit_preshower(compact_file, particle_name)
    elif geo_config == "linear_ratio":
        return _get_mem_limit_linear_ratio(compact_file, particle_name)
    else:
        return _get_mem_limit_basic(compact_file, particle_name)


def _get_mem_limit_basic(compact_file, particle_name):
    """
    Memory limits for the basic (num_layers × steel_ratio) MOBO parameter space.

    load = scint_mm × num_layers  — proxy for photon yield × detector volume.
    Tiers derived from test_memory_limits.py results (max_observed × 1.25, rounded up).
    """
    scint  = get_scint_thickness_mm(compact_file)
    layers = get_num_layers_from_xml(compact_file)
    scint  = scint  if scint  is not None else 20.0
    layers = layers if layers is not None else 14
    load   = scint * layers

    if particle_name in ("pi+", "kaon0L", "proton"):
        if scint <= 30.0:                        # thin scint
            if   load > 450: return "20G"        # boundary scint (~30mm) + many layers → use thick tier
            return "12G" if layers > 12 else "8G"
        else:                                    # thick scint
            if   load > 750: return "26G"        # worst 20.48 GB → ×1.25=25.6 → 26G
            elif load > 450: return "20G"        # worst 15.79 GB → ×1.25=19.7 → 20G
            else:            return "16G"        # worst 12.55 GB (scint=46.96mm, 9L) → ×1.25=15.7 → 16G
    elif particle_name == "neutron":
        if scint <= 30.0:
            if   load > 450: return "16G"        # boundary scint (~30mm) + many layers → use thick tier
            return "7G"                          # worst 5.63 GB → ×1.25=7.0 → 7G
        else:
            if   load > 750: return "18G"        # worst 14.72 GB → ×1.25=18.4 → 18G
            elif load > 450: return "16G"        # worst 11.97 GB (scint=53.75mm, 13L) → ×1.25=15.0 → 16G
            else:            return "10G"        # worst  7.60 GB (scint=44.16mm,  9L) → ×1.25=9.5 → 10G
    else:                                        # mu-
        if scint <= 30.0:
            if   load > 450: return "8G"         # boundary scint (~30mm) + many layers → use thick tier
            return "5G"                          # worst 4.08 GB → ×1.25=5.1 → 5G
        else:
            return "11G" if load > 750 else "8G" # >750: 8.63→×1.25=10.8→11G, ≤750: 6.04→×1.25=7.6→8G


def _get_mem_limit_preshower(compact_file, particle_name):
    """
    Memory limits for the preshower (preshower_steel_value × division_layer_number)
    MOBO parameter space.

    Validated by test_memory_limits_preshower.py (2026-03-06), 16 geometries ×
    3 particles × 3 reps across full parameter space (preshower_steel 10–101mm,
    division_layer_number 1–7). Memory is nearly flat across the space — no
    tiering needed.

    Worst-case observed at 7div, preshower_steel=10mm (max divisions, min absorption):
      pi+     : 7.97 GB → ×1.25 = 9.96 → 12G
      neutron : 5.80 GB → ×1.25 = 7.25 →  8G
      mu-     : 3.36 GB → ×1.25 = 4.20 →  5G
    """
    if particle_name in ("pi+", "kaon0L", "proton"):
        return "10G"   # worst 7.97 GB (7div, ps010, N=9) → ×1.25=9.96 → 10G
    elif particle_name == "neutron":
        return "8G"    # worst 5.80 GB (7div, ps010) → ×1.25=7.25 → 8G; was 7G (17% headroom)
    else:              # mu-
        return "5G"    # worst 3.36 GB (5div, ps037) → ×1.25=4.20 → 5G
    

def _get_mem_limit_linear_ratio(compact_file, particle_name):
    """
    Memory limits for the linear_ratio (steel_slope × scint_slope × steel_ratio)
    MOBO parameter space.

    From KLMWS_geo.cpp (linear_ratio branch): steel_slope and scint_slope ARE read
    and apply a linear per-layer thickness ramp:
      s_thick = s_thick_orig * (1 - slope + (layer-1) * 2*slope / (N-1))
    Total scintillator/steel volume is conserved (slope averages to 1×), but the
    distribution across layers affects absorption: high positive steel_slope →
    very thin steel in early layers → more particle throughput → more hits → more RAM.

    Primary memory driver: HcalSteelThickness = 75.5 * steel_ratio (mm).
    Secondary effect:      steel_slope (thin early steel slightly increases hits).

    Memory tiers are based on HcalSteelThickness, chosen to cover the worst slope
    within each tier.  Limits are conservative placeholders — update after running
    test_memory_limits_linear_ratio.py.

    steel_ratio range: 0.3–0.9 → HcalSteelThickness = 22.65mm–67.95mm

    Tiers:
      thin steel  (< 35mm, steel_ratio < 0.46): minimum absorption, worst case
      thick steel (≥ 35mm, steel_ratio ≥ 0.46): near-full absorption, fewer hits

    NOTE: These are conservative initial values. Run test_memory_limits_linear_ratio.py
    to determine validated limits from actual MaxRSS measurements.
    """
    steel_mm = get_steel_thickness_mm(compact_file)
    steel_mm = steel_mm if steel_mm is not None else 55.5  # default baseline

    if particle_name in ("pi+", "kaon0L", "proton"):
        if steel_mm < 35.0:
            return "25G"   # thin (ratio 0.3–0.46): max observed 19.62 GB × 1.25
        else:
            return "16G"   # thick (ratio 0.46–0.9): max observed 12.06 GB × 1.25
    elif particle_name == "neutron":
        if steel_mm < 35.0:
            return "17G"   # max observed 12.85 GB × 1.25
        else:
            return "12G"   # max observed 8.88 GB × 1.25
    else:                  # mu-
        if steel_mm < 35.0:
            return "9G"    # max observed 7.03 GB × 1.25
        else:
            return "7G"    # max observed 5.34 GB × 1.25


def describe_geometry(compact_file, geo_config="basic"):
    """Return a human-readable string describing the geometry parameters used for mem limit."""
    if geo_config == "preshower":
        div = get_division_layer_from_xml(compact_file)
        ps  = get_preshower_steel_from_xml(compact_file)
        return f"division_layer_number={div}, preshower_steel={ps}mm"
    elif geo_config == "linear_ratio":
        steel  = get_steel_thickness_mm(compact_file)
        scint  = get_scint_thickness_mm(compact_file)
        layers = get_num_layers_from_xml(compact_file)
        ratio  = round(steel / (steel + scint), 3) if steel and scint else None
        return f"HcalSteel={steel}mm, HcalScint={scint}mm, steel_ratio≈{ratio}, layers={layers}"
    else:
        scint  = get_scint_thickness_mm(compact_file)
        layers = get_num_layers_from_xml(compact_file)
        return f"scint={scint}mm, layers={layers}"
