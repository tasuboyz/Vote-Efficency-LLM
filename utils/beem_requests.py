import time
import random
import requests
from beem import Steem, Hive
from settings.config import HIVE_NODES, STEEM_NODES, BLOCKCHAIN_CHOICE, MAX_RESULTS
from settings.logging_config import logger
from beem.account import Account

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

def get_account_history(account, blockchain, max_retries=3, delay=1):
    """Get account history with retry logic for node failures and node switching."""
    for attempt in range(max_retries):
        try:
            history_data = []
            curation_count = 0
            
            for h in account.history_reverse():
                if h['type'] == 'curation_reward':
                    history_data.append(h)
                    curation_count += 1
                    if curation_count >= MAX_RESULTS:
                        logger.info(f"Collected {curation_count} curation rewards")
                        return history_data, blockchain
                        
            logger.info(f"Collected all available curation rewards: {curation_count}")
            return history_data, blockchain
            
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to get account history after {max_retries} attempts: {e}")
                raise
            
            logger.warning(f"Failed to get account history, retrying with new node... ({attempt + 1}/{max_retries})")
            time.sleep(delay)
            
            try:
                current_node = account.blockchain.rpc.url
                nodes = HIVE_NODES if BLOCKCHAIN_CHOICE == "HIVE" else STEEM_NODES
                working_nodes = [node for node in nodes if node != current_node and test_node(node)]
                
                if not working_nodes:
                    logger.error("No alternative working nodes found")
                    continue
                
                new_node = random.choice(working_nodes)
                new_blockchain = Hive(node=new_node) if BLOCKCHAIN_CHOICE == "HIVE" else Steem(node=new_node)
                new_account = Account(account.name, blockchain_instance=new_blockchain)
                
                logger.info(f"Switched to new node: {new_node}")
                account = new_account
                blockchain = new_blockchain
                
            except Exception as node_error:
                logger.warning(f"Failed to switch node: {str(node_error)}")
                continue
    
    raise Exception(f"Failed to get account history after {max_retries} attempts with different nodes")