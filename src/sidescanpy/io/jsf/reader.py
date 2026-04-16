'''JSF file reading and parsing logic

2026 rshomberg@uri.edu
'''

from pathlib import Path
from typing import List, Dict
import logging

from .messages import JSFMessageHeader, JSFMessageBody

logger = logging.getLogger(__name__)

def parse_jsf_file(fpath: Path) -> list[JSFMessageBody]:
    """
    Parse all messages from a JSF file.
    
    Args:
        fpath: Path to JSF file
        
    Returns:
        List of parsed JSF messages
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} does not exist!")

    with open(fpath, 'rb') as f:
        logger.info(f"Opening: {fpath}")
        all_messages = []
        
        while True:
            header_bytes = f.read(JSFMessageHeader.STRUCT.size)
            if len(header_bytes) < JSFMessageHeader.STRUCT.size:
                break  # EOF or truncated message
            
            header = JSFMessageHeader.from_bytes(header_bytes)
            logger.debug(f"{len(all_messages)+1} - {header}")

            message_bytes = f.read(header.message_size)
            try:
                message = JSFMessageBody.from_bytes(header.message_type, message_bytes)
                message.set_context(header)
                all_messages.append(message)
                logger.debug(f"{len(all_messages)} - {message}")
            except ValueError as e:
                logger.debug(e)
                return all_messages
            
    return all_messages


def split_messages(messages: list[JSFMessageBody]) -> dict[str, list[JSFMessageBody]]:
    """
    Classify messages by type.
    
    Args:
        messages: List of JSF messages
        
    Returns:
        Dictionary mapping message type names to lists of messages
    """
    groups = {}  # Auto-generate groups from message classes
    
    for msg in messages:
        # Get class name and pluralize it
        cls_name = type(msg).__name__ + 's'
        
        if cls_name not in groups:
            groups[cls_name] = []
        
        groups[cls_name].append(msg)
    
    return groups


if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Parse JSF file')
    parser.add_argument('jsf_file', type=Path, help='Path to JSF file')
    parser.add_argument('--split', action='store_true', help='Split and display message groups')
    parser.add_argument('--head', type=int, default=10, help='Show first N sonar messages')
    args = parser.parse_args()
    
    # Parse file
    messages = parse_jsf_file(args.jsf_file)
    logger.info(f"Parsed {len(messages)} total messages")
    
    if args.split:
        groups = split_messages(messages)
        logger.info("Message breakdown:")
        for msg_type in sorted(groups.keys()):
            count = len(groups[msg_type])
            logger.info(f"  {msg_type}: {count}")
        
        # Show sample sonar messages
        if 'SonarDataMessages' in groups:
            logger.info(f"\nFirst {args.head} SonarDataMessages:")
            for msg in groups['SonarDataMessages'][:args.head]:
                logger.info(f"  {msg}")