from typing import Union, Optional

import ioc_finder
import ioc_fanger
from libs.iocp import Parser

import pandas as pd
from tqdm import tqdm

from multiprocessing import Pool
from itertools import chain

parser = Parser.Parser()

ioc_key_map = {
    "Filepath": 'path',
    "file_paths": 'path',
    "Host": 'domain',
    "domains": 'domain',
    "Filename": 'file',
    "Registry": 'registry-key',
    "registry_key_paths": 'registry-key',
    "CVE": 'cve',
    "cves": 'cve',
    "ipv4s": 'ip-address',
    "ipv4_cidrs": 'ip-address',
    "ipv6s": 'ip_address',
    "asns": 'asn',
    "SHA256": 'hash',
    "sha256s": 'hash',
    "sha1s": 'hash',
    "SHA1": 'hash',
    "md5s": 'hash',
    "MD5": 'hash',
    "ssdeeps": 'hash',
    "urls": 'url',
    "URL": 'url',
    "enterprise-techniques": 'mitre-attack',
    "bitcoin_addresses": 'bitcoin_addresse',
    "email_addresses_complete": 'email-address',
    "email_addresses": 'email-address',
    "user_agents": 'user-agent',
    "mac_addresses": 'mac-address'
}


def flatten_ioc(iocs: dict) -> list[tuple]:
    """
    Extract types of the IOCs exist in the extraction from ioc-finder

    :param iocs: Extracted IOC mapping (`tuple`)
    :return: `list` containing existing IOCs
    """
    ioc_types = []
    for ioc_type, values in iocs.items():
        if isinstance(values, str):
            ioc_types.append((ioc_type, values))
        elif isinstance(values, list) and len(values) > 0:
            for value in values:
                ioc_types.append((ioc_type, value))
        elif isinstance(values, dict):
            ioc_types.extend(
                [(f"{i_type}-{ioc_type.split('_')[-1]}", i_value) for i_type, i_value in flatten_ioc(values)])
    return ioc_types


def get_ioc_types(iocs):
    """
    Extract types of the IOCs exist in the extraction from ioc-finder

    :param iocs: Extracted IOC mapping (`dict`)
    :return: `list` containing existing IOCs
    """
    ioc_types = []
    for ioc_type, values in iocs.items():
        if isinstance(values, list) and len(values) > 0:
            ioc_types.append(ioc_type)
        if isinstance(values, dict):
            ioc_types.extend(get_ioc_types(values))
    return ioc_types


def extract_from_record(record: dict, text_col: str = 'text') -> list[dict]:
    """
    Extract IoCs from a single record

    :param record: Record (`dict`) containing `text_col` key.
    :param text_col: Key of the record containing the text with IoCs
    :return: `list` of records expanded with IoCs
    """
    out = []
    if isinstance(record[text_col], str):
        # Extract with ioc-finder
        fanged_text = ioc_fanger.fang(record[text_col])
        iocs_f = ioc_finder.find_iocs(fanged_text)
        iocs_f = flatten_ioc(iocs_f)

        # Extract with iocp
        iocs_p = parser.parse_str(record[text_col])
        iocs_p = flatten_ioc(iocs_p)

        unique = {}
        for ioc_type, ioc_value in (iocs_f + iocs_p):
            if ioc_value in unique:
                continue
            out.append({**record, 'ioc': ioc_key_map[ioc_type], 'ioc_value': ioc_value})
            unique[ioc_value] = True
    return out


def extract_iocs(records: list[dict], text_col: str = 'text', num_workers: Optional[int] = None, progress: bool = True):
    """
    Extract IoCs from a list of records

    :param record: Records (`list[dict]`) where each record containing `text_col` key.
    :param text_col: Key of the record containing the text with IoCs
    :param num_workers: Number of workers to multiprocess. Serial if set to `None`.
    :param progress: Whether to print the progress or not.
    :return: `list` of records expanded with IoCs
    """
    if num_workers is not None:
        with Pool(processes=num_workers) as pool:
            results = [x for x in
                       tqdm(pool.imap(extract_from_record, records), total=len(records), disable=not progress)]
        return chain.from_iterable(results)
    else:
        results = [extract_from_record(record) for record in tqdm(records, total=len(records))]
        return chain.from_iterable(results)


def sanitize_iocs(sentences: Union[list[dict], pd.DataFrame], iocs: Optional[Union[list[dict], pd.DataFrame]] = None,
                  text_col: str = 'text', num_workers: Optional[int] = None, progress: bool = True) -> Union[
    list[dict], pd.DataFrame]:
    """
    Santize IoCs from sentences

    :param sentences: Sentence records (`list[dict]`) or dataframe (`pd.DataFrame`) where each record containing `text_col` key.
    :param iocs: IoCs records (`list[dict]`) or dataframe (`pd.DataFrame`) where each record containing `text_col` key.
    :param text_col: Key of the record containing the text with IoCs
    :param num_workers: Number of workers to multiprocess. Serial if set to `None`.
    :param progress: Whether to print the progress or not.
    :return: Sentence records (`list[dict]`) or dataframe (`pd.DataFrame`) with added `clean` texts
    """
    if iocs is None:
        records = sentences if isinstance(sentences, list) else sentences.to_dict(orient="records")
        iocs = pd.DataFrame(extract_iocs(records, text_col, num_workers, progress))
    elif isinstance(iocs, list):
        iocs = pd.DataFrame(iocs)

    sanitized_sent = []
    for (text, group) in iocs.groupby(text_col):
        cleaned = text
        for i_type, i_value in group.loc[:, ['ioc', 'ioc_value']].values:
            cleaned = cleaned.replace(i_value, i_type)
        sanitized_sent.append({text_col: text, 'clean': cleaned})
    sanitized_sent = pd.DataFrame(sanitized_sent)

    was_records = False
    if isinstance(sentences, list):
        sentences = pd.DataFrame(sentences)
        was_records = True

    sanitized_sent = sentences.join(sanitized_sent.set_index(text_col), on=text_col)
    sanitized_sent['sanitized'] = ~sanitized_sent.clean.isna()
    sanitized_sent['clean'] = sanitized_sent.apply(
        lambda x: x['clean'] if isinstance(x['clean'], str) else x['text'], axis=1, result_type='reduce'
    )
    return sanitized_sent if not was_records else sanitized_sent.to_dict(orient='records')
