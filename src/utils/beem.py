import logging
import random
import requests
from beem import Steem, Hive
from beem.account import Account
from beem.vote import Vote
from beem.comment import Comment
from ..settings.config import STEEM_NODES, HIVE_NODES
from ..settings.logger_config import logger

def convert_vests_to_power(amount, blockchain_instance):
    """Convert vesting shares to HP/SP based on blockchain type."""
    try:
        if isinstance(blockchain_instance, Hive):
            return blockchain_instance.vests_to_hp(float(amount))
        elif isinstance(blockchain_instance, Steem):
            return blockchain_instance.vests_to_sp(float(amount))
        else:
            logger.error("Unsupported blockchain")
            return 0
    except Exception as e:
        logger.error(f"Error converting vesting shares to power: {e}")
        return 0

def test_node(node_url, blockchain_type="HIVE"):
    """Test if a node is responsive and functioning correctly."""
    try:
        headers = {'Content-Type': 'application/json'}
        payload = {
            "jsonrpc": "2.0",
            "method": "condenser_api.get_dynamic_global_properties",
            "params": [],
            "id": 1
        }
        
        response = requests.post(node_url, json=payload, headers=headers, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            if 'result' in result:
                logger.info(f"Node {node_url} is working properly")
                return True
            
        logger.warning(f"Node {node_url} returned invalid response")
        return False
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Node {node_url} test failed: {str(e)}")
        return False

def get_working_node(blockchain_type="HIVE"):
    """Get a working node from the configured list with fallback mechanism."""
    nodes = HIVE_NODES if blockchain_type == "HIVE" else STEEM_NODES
    working_nodes = []
    
    for node in nodes:
        if test_node(node, blockchain_type):
            working_nodes.append(node)
    
    if not working_nodes:
        raise Exception(f"No working {blockchain_type} nodes found!")
    
    selected_node = random.choice(working_nodes)
    logger.info(f"Selected {blockchain_type} node: {selected_node}")
    return selected_node

def switch_to_backup_node(current_node, blockchain_type="HIVE"):
    """Switch to a different working node, avoiding the current one."""
    nodes = HIVE_NODES if blockchain_type == "HIVE" else STEEM_NODES
    backup_nodes = [node for node in nodes if node != current_node]
    
    for node in backup_nodes:
        if test_node(node, blockchain_type):
            logger.info(f"Switching from {current_node} to backup node: {node}")
            return node
    
    raise Exception(f"No working backup {blockchain_type} nodes available!")

def initialize_blockchain(blockchain_choice="HIVE"):
    """Initialize blockchain connection with automatic node selection."""
    try:
        working_node = get_working_node(blockchain_choice)
        if blockchain_choice == "HIVE":
            blockchain = Hive(node=working_node)
            power_symbol = "HP"
        else:
            blockchain = Steem(node=working_node)
            power_symbol = "SP"
        
        logger.info(f"Connected to {blockchain_choice} node: {working_node}")
        return blockchain, power_symbol, working_node
    
    except Exception as e:
        logger.error(f"Failed to initialize blockchain connection: {str(e)}")
        raise

def get_post_data(post_identifier, blockchain_instance):
    """Get post data from blockchain."""
    return Comment(post_identifier, blockchain_instance=blockchain_instance)

def get_vote_data(vote_identifier, blockchain_instance):
    """Get vote data from blockchain."""
    return Vote(vote_identifier, blockchain_instance=blockchain_instance)

def get_account(account_name, blockchain_instance):
    """Get account data from blockchain."""
    return Account(account_name, blockchain_instance=blockchain_instance)