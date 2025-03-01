# Vote Efficiency ML Bot

This is a machine learning bot designed to optimize voting efficiency on STEEM and HIVE blockchains.

## Configuration Setup

### Required Files

1. Create `keys.py` in the `settings` folder:

```python
# filepath: /settings/keys.py
steem_posting_key = "your_posting_key_here"
hive_posting_key = "your_posting_key_here"
```

2. Create `config.json` in the root folder:

```json
{
    "admin_id": "YOUR TELEGRAM ID",
    "TOKEN": "TOKEN",
    "steem_curator": "tasuboyz",
    "hive_curator": "menny.trx",
}
```

### Configuration Details

- `admin_id`: Your Telegram user ID (numeric)
- `TOKEN`: Telegram bot token from BotFather
- `steem_curator`: Your STEEM blockchain username
- `hive_curator`: Your HIVE blockchain username
- `posting_key_steem`: STEEM blockchain posting key
- `posting_key_hive`: HIVE blockchain posting key

⚠️ **Security Notice**: 
- Never share your private keys
- Add `keys.py` and `config.json` to `.gitignore`
- Keep your configuration files secure

## Directory Structure

```
Vote-Efficency-ML/
├── settings/
│   ├── config.py
│   ├── keys.py (you need to create this)
│   └── logging_config.py
├── config.json (you need to create this)
└── ...
```