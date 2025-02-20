import logging

log_level = logging.INFO

# Manual node configuration
STEEM_NODES = [
    "https://api.steemit.com",
    "https://api.justyy.com",
    "https://api.moecki.online"
]

HIVE_NODES = [
    "https://api.deathwing.me",
    "https://api.hive.blog",
    "https://api.openhive.network",
]

CURATOR = "cur8"  # Account

BLOCKCHAIN_CHOICE = "HIVE"  # <-- Modificare qui per cambiare blockchain

log_file_path = 'log.txt'
