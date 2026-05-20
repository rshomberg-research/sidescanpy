'''JSF file reading and parsing logic

2026 rshomberg@uri.edu
'''

from pathlib import Path
# from typing import List, Dict
import logging

from sidescanpy.io.jsf.messages import JSFMessageHeader, JSFMessageBody
# Using import because file is way too long. It may make sense to combine files

def read_jsf(fpath: Path, N: int| None = None) -> list[JSFMessageBody]:
    """
    Parse all messages from a JSF file.
    """
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} does not exist!")

    with open(fpath, 'rb') as f:
        logging.info(f"Opening: {fpath}")
        all_messages = []
        cnt = 0
        while (N==None or cnt<N):
            header_bytes = f.read(JSFMessageHeader.STRUCT.size)
            if len(header_bytes) < JSFMessageHeader.STRUCT.size:
                break  # EOF or truncated message
            
            header = JSFMessageHeader.from_bytes(header_bytes)
            logging.debug(f"{len(all_messages)+1} - {header}")

            message_bytes = f.read(header.message_size)
            try:
                message = JSFMessageBody.from_bytes(header.message_type, message_bytes)
                message.set_context(header)
                all_messages.append(message)
                logging.debug(f"{len(all_messages)} - {message}")
            except ValueError as e:
                logging.debug(e)
                return all_messages
            cnt += 1
            
    return all_messages


def read_jsf_iter(fpath: Path) -> list[JSFMessageBody]: # TODO: generator version for memory efficiency
    ... 

def save_temp_file(data: list[JSFMessageBody], filepath: Path = Path("temp_data.pkl")) -> None:
    '''Primarily used for testing so I do not have to read the entire file again'''
    from pickle import dump
    with open(filepath, 'wb') as f:
        logging.debug(f"Saving {len(data)} messages to {filepath}")
        dump(data, f)

def load_temp_file(filepath: Path = Path("temp_data.pkl")) -> list[JSFMessageBody]:
    '''Primarily used for testing so I do not have to read the entire file again'''
    from pickle import load
    with open(filepath, 'rb') as f:
        logging.debug(f"Loading messages from {filepath}")
        data = load(f)
    logging.debug(f"Loaded {len(data)} messages")
    return data

# TODO: save/load temp file either a list or a dict

if __name__ == '__main__':
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # import argparse
    # parser = argparse.ArgumentParser(description='Parse JSF file')
    # parser.add_argument('jsf_file', type=Path, help='Path to JSF file')
    # parser.add_argument('--split', action='store_true', help='Split and display message groups')
    # parser.add_argument('--head', type=int, default=10, help='Show first N sonar messages')
    # args = parser.parse_args()
    # test_file = args.jsf_file

    # TODO: write actual tests and then use argparser for the standalone version
    jsf_file = Path('data/survey-a/jsf/20251015170335.jsf')
    head = 1000 # Limit for speed during tests
    # head = None

    # Parse file
    messages = read_jsf(jsf_file, N=head)
    logging.info(f"Parsed {len(messages)} total messages")

    fpath = Path("temp_jsf_data.pkl")
    save_temp_file(messages, filepath=fpath)
    logging.info(f"Saved to: {fpath}")
    messages = load_temp_file(fpath)
    logging.info(f"Loadded from: {fpath}")
    # TODO: delete temp file
    
