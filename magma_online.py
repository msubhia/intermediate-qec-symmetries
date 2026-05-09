# Helpers on top of autqec to run magma online
# parsing taken from autqec

import re
import numpy as np

from urllib.parse       import urlencode
from http.client        import HTTPSConnection
from xml.dom.minidom    import parseString

def run_magma_online_from_file(command_file):
    match = re.search(r"n(\d+)k(\d+)d(\d+)", command_file)
    if not match:
        raise ValueError(f"Could not parse n,k,d from {command_file}")

    n, k, d = match.groups()
    output_file = f"magma_output_n{n}k{k}d{d}.txt"

    with open(command_file, "r") as f:
        code = f.read()

    code = "SetColumns(0);\n" + code

    params = urlencode({"input": code})
    headers = {
        "Content-type": "application/x-www-form-urlencoded",
        "Accept": "application/xml,text/xml,*/*",
        "Referer": "https://magma.maths.usyd.edu.au/calc/",
        "User-Agent": "Mozilla/5.0",
    }

    conn = HTTPSConnection("magma.maths.usyd.edu.au", timeout=120)
    conn.request("POST", "/xml/calculator.xml", params, headers)
    response = conn.getresponse()

    raw = response.read()
    status = response.status
    content_type = response.getheader("Content-Type")
    conn.close()

    # Save raw response for debugging
    with open("magma_raw_response.txt", "wb") as f:
        f.write(raw)

    if status != 200:
        raise RuntimeError("Magma server did not return HTTP 200.")

    # Check if response is actually XML
    stripped = raw.lstrip()
    if not stripped.startswith(b"<?xml") and not stripped.startswith(b"<results"):
        raise RuntimeError(
            "Magma server returned non-XML. "
            "Check magma_raw_response.txt to see the actual response."
        )

    xml_doc = parseString(raw)

    result_lines = []

    for results in xml_doc.getElementsByTagName("results"):
        for line in results.getElementsByTagName("line"):
            text = "".join(
                node.data
                for node in line.childNodes
                if node.nodeType == node.TEXT_NODE
            )
            result_lines.append(text)

    output = "\n".join(result_lines)

    with open(output_file, "w") as f:
        f.write(output)


def parse_magma_output(magma_output_filename, qec_code_auts_from_magma_with_intersection_obj):
    with open(magma_output_filename, "r") as file:
        raw_magma_output = file.read()

    # time
    time_pattern = r"Time:\s*([\d\.]+)"
    match = re.search(time_pattern, raw_magma_output)
    if match:
        time = float(match.group(1))
    else:
        time = 0.0
    
    # automorphism group order
    order_pattern = r"Order:\s*(\d+)"
    match = re.search(order_pattern, raw_magma_output)
    if match:
        order = int(match.group(1))
    else:
        order = 1

    # automorphism group generators and qubit relabelling to original basis
    reordered_qubit_list, H_rref, transform_rows, transform_cols = qec_code_auts_from_magma_with_intersection_obj.preprocess_H_3bit()
    aut_gens, aut_gens_text = qec_code_auts_from_magma_with_intersection_obj.parse_magma_output_for_aut_gens(raw_magma_output)
    fixed_auts_gens = []
    for g in aut_gens:
        correct_g = []
        for cycle in g:
            new_cycle = []
            for i in cycle:
                new_cycle.append(reordered_qubit_list[i-1])
            correct_g.append(tuple(new_cycle))
        fixed_auts_gens.append(correct_g)

    # store in dictionary
    code_auts_dict = {}
    code_auts_dict['order'] = order
    code_auts_dict['auts'] = fixed_auts_gens
    code_auts_dict['time'] = time

    return code_auts_dict


