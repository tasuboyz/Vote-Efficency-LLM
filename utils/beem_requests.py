import time
import random
import requests
from beem import Steem, Hive
from beem.account import Account
from settings.config import HIVE_NODES, STEEM_NODES, BLOCKCHAIN_CHOICE, MAX_RESULTS
from settings.logging_config import logger
from beem.comment import Comment
from settings.keys import steem_posting_key, hive_posting_key
import json 

class BlockchainConnector:
    def __init__(self, blockchain_type="HIVE"):
        """Initialize blockchain connector with specified type."""
        self.blockchain_type = blockchain_type
        self.nodes = HIVE_NODES if blockchain_type == "HIVE" else STEEM_NODES
        self.working_node = self.get_working_node()
        self.blockchain = self._initialize_blockchain()
        self.power_symbol = "HP" if blockchain_type == "HIVE" else "SP"

    def _initialize_blockchain(self):
        """Initialize blockchain instance with working node."""
        return Hive(keys=[hive_posting_key], node=self.working_node) if self.blockchain_type == "HIVE" else Steem(keys=[steem_posting_key], node=self.working_node)

    def convert_vests_to_power(self, amount):
        """Convert vesting shares to HP/SP based on blockchain type."""
        try:
            if isinstance(self.blockchain, Hive):
                return self.blockchain.vests_to_hp(float(amount))
            elif isinstance(self.blockchain, Steem):
                return self.blockchain.vests_to_sp(float(amount))
            else:
                logger.error("Unsupported blockchain")
                return 0
        except Exception as e:
            logger.error(f"Error converting vesting shares to power: {e}")
            return 0

    def test_node(self, node_url):
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

    def get_working_node(self):
        """Get a working node from the configured list with fallback mechanism."""
        working_nodes = []
        
        for node in self.nodes:
            if self.test_node(node):
                working_nodes.append(node)
        
        if not working_nodes:
            raise Exception(f"No working {self.blockchain_type} nodes found!")
        
        selected_node = random.choice(working_nodes)
        logger.info(f"Selected {self.blockchain_type} node: {selected_node}")
        return selected_node

    def switch_to_backup_node(self):
        """Switch to a different working node, avoiding the current one."""
        backup_nodes = [node for node in self.nodes if node != self.working_node]
        random.shuffle(backup_nodes)  # Randomize the order of backup nodes
        working_backup_nodes = []
        
        # First collect all working backup nodes
        for node in backup_nodes:
            if self.test_node(node):
                working_backup_nodes.append(node)
        
        if not working_backup_nodes:
            raise Exception(f"No working backup {self.blockchain_type} nodes available!")
        
        # Select a random working backup node
        new_node = random.choice(working_backup_nodes)
        logger.info(f"Switching from {self.working_node} to backup node: {new_node}")
        self.working_node = new_node
        self.blockchain = self._initialize_blockchain()
        return True

    def get_account_history(self, account_name, max_retries=3, delay=1):
        """Get account history with retry logic for node failures and node switching."""
        # Create a list of all available nodes except current one
        available_nodes = [node for node in self.nodes if node != self.working_node]
        # Add current node at the beginning
        all_nodes = [self.working_node] + available_nodes
        
        for node in all_nodes:
            try:
                # Switch to new node
                self.working_node = node
                self.blockchain = self._initialize_blockchain()
                account = Account(account_name, blockchain_instance=self.blockchain)
                logger.info(f"Trying node: {node}")
                
                history_data = []
                curation_count = 0
                
                for h in account.history_reverse():
                    if h['type'] == 'curation_reward':
                        history_data.append(h)
                        curation_count += 1
                        if curation_count >= MAX_RESULTS:
                            logger.info(f"Collected {curation_count} curation rewards from {node}")
                            return history_data, self.blockchain
                
                logger.info(f"Collected all available curation rewards: {curation_count} from {node}")
                return history_data, self.blockchain
                
            except Exception as e:
                logger.warning(f"Failed to get account history from node {node}: {str(e)}")
                continue
        
        raise Exception(f"Failed to get account history after trying all nodes")
    
    def get_author_post(self, author, platform):
        data = {
            "jsonrpc": "2.0",
            "method": "condenser_api.get_discussions_by_blog",
            "params": [{"tag": author, "limit": 1}],
            "id": 1
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self.working_node, headers=headers, data=json.dumps(data), timeout=5)
        response.raise_for_status()
        result = response.json().get('result', [])
        return result[0]
    
    def get_account_info(self, account_name):
        return Account(account_name, blockchain_instance=self.blockchain)
    
    def calculate_voting_power(self, account_name):
        accout = Account(account_name, blockchain_instance=self.blockchain)
        return accout.get_voting_power()
    
    def get_permlink(self, post_url):
        comment = Comment(post_url, blockchain_instance=self.blockchain)
        permlink = comment.permlink
        return permlink
    
    def get_author(self, post_url):
        comment = Comment(post_url, blockchain_instance=self.blockchain)
        author = comment.author
        return author
    
    def like_steem_post(self, voter, voted, permlink, weight=20):

        account = Account(voter, blockchain_instance=self.blockchain)
        comment = Comment(authorperm=f"@{voted}/{permlink}", blockchain_instance=self.blockchain)
        comment.vote(weight, account=account)