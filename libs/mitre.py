import os
import json
import pandas as pd


default_data_dir = '/home/local/QCRI/ukumarasinghe/projects/CTI2AttackMetrix/data/mitre'


def list_tactics(data_dir=default_data_dir):
    """
    List MITRE tactics

    :param data_dir: Directory of scraped data
    :return: `list` of tactics
    """
    scapred_path = os.path.join(data_dir, 'scraped')
    return [os.path.join(scapred_path, f) for f in os.listdir(scraped_path) if '.json' in f and 'TA' in f]


def list_techniques(data_dir=default_data_dir):
    """
    List MITRE techniques

    :param data_dir: Directory of scraped data
    :return: `list` of techniques
    """
    scapred_path = os.path.join(data_dir, 'scraped')
    return [os.path.join(scapred_path, f) for f in os.listdir(scapred_path) if '.json' in f and not 'TA' in f]


def load_technique_map(data_dir=default_data_dir, full_name=False):
    """
    Load mapping of a MITRE technique IDs to technique names

    :param data_dir: Directory of scraped data
    :param full_name: Include the parent technique name in sub-techniques
    :return: `dict` with contents of MITRE techinque mapping
    """    
    techniques = list_techniques(data_dir)

    tech_map = {}
    for tech_file in techniques:
        tech = load_technique_file(tech_file)
        tech_map[tech['ID']] = tech['Name']

    tech_map['T1521'] = 'Encrypted Channel'
    tech_map['T1533'] = 'Data from Local System'
    tech_map['T1218'] = 'System Binary Proxy Execution'
    tech_map['T1053.001'] = 'At'
    
    if full_name:
        for k, v in tech_map.items():
            tech_map[k] = v if len(k) == 5 else f"{tech_map[k[:5]]}: {v}"

    return tech_map

    
def load_technique(tech_id, data_dir=default_data_dir):
    """
    Load resources of a MITRE technique

    :param tech_id: ID of the MITRE technique
    :param data_dir: Directory of scraped data
    :return: `dict` with contents of MITRE resource
    """    
    return load_technique_file(os.path.join(data_dir, f'{tech_id}.json'), 'r')


def load_technique_file(file_path):
    """
    Load resources of a MITRE technique

    :param filepath: Path to file containing MITRE technique
    :return: `dict` with contents of MITRE resource
    """
    with open(file_path, 'r') as f:
        technique = json.load(f)

    return technique


def load_sources(data_dir=default_data_dir):
    """
    Load sources referenced in MITRE

    :param data_dir: Directory of scraped data
    :return: `pd.DataFrame` containing MITRE references
    """    
    return pd.read_csv(os.path.join(data_dir, 'meta_references.csv'))
