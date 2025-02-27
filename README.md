# Vote Efficiency ML Bot

This is a machine learning bot designed to optimize voting efficiency on STEEM and HIVE blockchains.

## Configuration Setup

### Required Files

1. Create `keys.py` in the `settings` folder:

```python
# filepath: /settings/keys.py
POSTING_KEY = "your_posting_key_here"
ACTIVE_KEY = "your_active_key_here"
```

2. Create `config.json` in the root folder:

```json
{
    "admin_id": "your_telegram_user_id",
    "TOKEN": "your_telegram_bot_token",
    "steem_curator": "your_steem_username",
    "hive_curator": "your_hive_username",
    "posting_key_steem": "your_steem_posting_key",
    "posting_key_hive": "your_hive_posting_key"
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