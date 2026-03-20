#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging


# # https://wis-tns.weizmann.ac.il/api/get
def format_to_json(source):

    import json
    from collections import OrderedDict

    """
    
    Change data to json format and return
    
    :param source: Dictionary
    :type source: dict
    :return:  data in json format 
    :rtype: json

    """
    parsed = json.loads(source, object_pairs_hook=OrderedDict)
    result = json.dumps(parsed, indent=4)

    return result


def plot_transient_info(data):
    """
    Nicely print key transient info from TNS.
    """
    if not data:
        logging.getLogger(__name__).info("No TNS data available to display.")
        return

    logger = logging.getLogger(__name__)
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
    import requests
    import json
    from collections import OrderedDict

    """
    Query the Transient Name Server (TNS) for transient coordinates and basic info.

    :param objname: IAU name of transient (e.g. '2024aifs'). Prefixes like 'SN' or 'AT' are stripped.
    :param TNS_BOT_ID: TNS bot ID
    :param TNS_BOT_NAME: TNS bot name
    :param TNS_BOT_API: TNS bot API key
    :return: Dictionary with transient data or None if not found
    """
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
            logging.getLogger(__name__).warning(
                "No TNS data returned for object '%s'.", objname
            )
            return None

        logging.getLogger(__name__).info("Found transient '%s' on TNS.", objname)
        plot_transient_info(data)
        return data

    except Exception as exc:
        logging.getLogger(__name__).error(
            "Error querying TNS for object '%s': %s", objname, exc, exc_info=True
        )
        return None
