from pathlib import Path
import logging

from ...io.jsf.messages import JSFMessageBody
from ...io.jsf.messages import JSFMessageHeader

from ...io.jsf.messages import parse_jsf_file as read_jsf_file

# FIXME: this is not used at all

# def read_jsf_file(path: Path) -> list[JSFMessageBody]:
#     logging.debug(f"Opening {path}")
#     messages = []
#     with open(path, "rb") as f:
#         try:
#             while True:
#                 msg = parse_message(f)
#                 messages.append(msg)
#         except EOFError:
#             pass

#     return messages

def read_jsf_dir(path: Path) -> list[JSFMessageBody]:
    all_msgs = []
    for file in sorted(Path(path).glob("*.jsf")):
        all_msgs.extend(read_jsf_file(file))
    return all_msgs

