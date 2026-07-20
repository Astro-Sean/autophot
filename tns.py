#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transient Name Server (TNS) query helpers for AutoPHOT.

Provides functions to query the `TNS API <https://www.wis-tns.org/api/get>`_
for transient coordinates, type, redshift, and discovery metadata.
Credentials are resolved via :mod:`autophot_tokens`.
"""

import json
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


def format_to_json(source):
    """Parse a JSON string and re-emit it with pretty-printing.

    Parameters
    ----------
    source : str
        Raw JSON text returned by the TNS API.

    Returns
    -------
    str
        Indented JSON string (key order preserved).
    """
    parsed = json.loads(source, object_pairs_hook=OrderedDict)
    return json.dumps(parsed, indent=4)


def plot_transient_info(data):
    """Log a human-readable summary of transient metadata from TNS.

    Parameters
    ----------
    data : dict or None
        TNS object data dictionary.  If ``None`` or empty, a message is
        logged and the function returns immediately.
    """
    if not data:
        logger.info("No TNS data available to display.")
        return

    summary_lines = [
        "Transient Summary:",
        "-" * 40,
        f"Name:         {data.get('name_prefix', '')}{data.get('objname', '')}",
        f"Type:         {data.get('object_type', {}).get('name', 'Unknown')}",
        f"RA / Dec:     {data.get('ra')} / {data.get('dec')}",
        f"RA / Dec deg: {data.get('radeg')} / {data.get('decdeg')}",
        f"Redshift:     {data.get('redshift', 'N/A')}",
        f"Discovered:   {data.get('discoverydate')}",
        (
            "Disc. Mag:    "
            f"{data.get('discoverymag')} mag "
            f"(filter: {data.get('discmagfilter', {}).get('name', 'N/A')})"
        ),
        f"Discoverer:   {data.get('discoverer')}",
        f"Reporter:     {data.get('reporter')}",
    ]
    internal = data.get("internal_names")
    if internal:
        summary_lines.append(f"Internal IDs: {internal}")
    summary_lines.append("-" * 40)

    for line in summary_lines:
        logger.info(line)


def get_coords(objname, TNS_BOT_ID=None, TNS_BOT_NAME=None, TNS_BOT_API=None):
    """Query the TNS for transient coordinates and basic metadata.

    Parameters
    ----------
    objname : str
        IAU name of the transient (e.g. ``'2024aifs'``).
        Prefixes like ``'SN'`` or ``'AT'`` are stripped automatically.
    TNS_BOT_ID : int, optional
        TNS bot ID.
    TNS_BOT_NAME : str, optional
        TNS bot name.
    TNS_BOT_API : str, optional
        TNS bot API key.

    Returns
    -------
    dict or None
        TNS object data dictionary, or ``None`` if the object is not found
        or the query fails.

    Raises
    ------
    ValueError
        If any of the three credential arguments are missing.
    """
    import requests

    if not all([TNS_BOT_ID, TNS_BOT_NAME, TNS_BOT_API]):
        raise ValueError("You must provide TNS_BOT_ID, TNS_BOT_NAME, and TNS_BOT_API")

    objname = objname.strip()
    if objname.lower().startswith(("sn", "at")):
        objname = objname[2:]

    url = "https://www.wis-tns.org/api/get/object"
    headers = {
        "User-Agent": f'tns_marker{{"tns_id": {TNS_BOT_ID}, "type": "bot", "name": "{TNS_BOT_NAME}"}}'
    }

    payload = {
        "api_key": TNS_BOT_API,
        "data": json.dumps(OrderedDict([("objname", objname)])),
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        json_data = json.loads(format_to_json(response.text))
        data = json_data.get("data")

        if not data:
            logger.warning("No TNS data returned for object '%s'.", objname)
            return None

        logger.info("Found transient '%s' on TNS.", objname)
        plot_transient_info(data)
        return data

    except Exception as exc:
        logger.error(
            "Error querying TNS for object '%s': %s", objname, exc, exc_info=True
        )
        return None
