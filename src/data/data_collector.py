from datetime import datetime, timezone
from typing import Dict, List, Optional
from beem.account import Account
from beem.comment import Comment
from beem.vote import Vote
from ..utils.decorators import retry_on_blockchain_failure, log_execution_time
from ..utils.beem import get_post_data, get_vote_data, convert_vests_to_power
from ..settings.logger_config import logger

class BlockchainDataCollector:
    def __init__(self, blockchain_instance, curator: str, power_symbol: str):
        self.blockchain = blockchain_instance
        self.curator = curator
        self.power_symbol = power_symbol
        self.data = self._initialize_data_structure()

    def _initialize_data_structure(self) -> Dict:
        """Initialize the data structure for collecting blockchain data."""
        return {
            'voting_power': [],
            'vote_delay': [],
            'reward': [],
            'author_avg_efficiency': [],
            'success': [],
            'author_reputation': [],
            'Post': [],
            'Author': [],
            'like_efficiency': [],
            'author_avg_payout': [],
        }

    @retry_on_blockchain_failure()
    @log_execution_time
    def collect_account_history(self, limit: int = 1000) -> Dict:
        """Collect account history data from blockchain."""
        account = Account(self.curator, blockchain_instance=self.blockchain)
        count = 0
        
        for h in account.history_reverse():
            if count >= limit:
                break
                
            if h['type'] == 'curation_reward':
                try:
                    self._process_curation_reward(h)
                    count += 1
                except Exception as e:
                    logger.error(f"Error processing curation reward: {e}")
                    continue
        
        return self.data

    def _process_curation_reward(self, history_item: Dict) -> None:
        """Process a single curation reward entry."""
        try:
            # Extract post identifiers
            author = history_item.get('comment_author') or history_item.get('author')
            permlink = history_item.get('comment_permlink') or history_item.get('permlink')
            post_identifier = f"@{author}/{permlink}"
            
            # Get blockchain data
            post_data = get_post_data(post_identifier, self.blockchain)
            vote_data = get_vote_data(f"{post_identifier}|{self.curator}", self.blockchain)
            
            # Process the data
            self._extract_and_store_metrics(history_item, post_data, vote_data)
            
        except Exception as e:
            logger.error(f"Failed to process curation reward: {str(e)}")
            raise

    def _extract_and_store_metrics(self, history_item: Dict, post_data: Comment, vote_data: Vote) -> None:
        """Extract metrics from blockchain data and store them."""
        try:
            # Calculate timestamps and delays
            post_time = post_data['created'].replace(tzinfo=timezone.utc)
            vote_time = vote_data.time.replace(tzinfo=timezone.utc)
            vote_delay = (vote_time - post_time).total_seconds() / 60  # Convert to minutes

            # Extract and normalize reward data
            reward_vests = float(history_item['reward']['amount']) / 1e6  # Convert from µVESTS to VESTS
            reward = convert_vests_to_power(reward_vests, self.blockchain)
            
            # Calculate vote weight and voting power
            vote_weight = float(vote_data.weight) / 1000000000 # Convert to percentage (0-100)
            teoric_reward = convert_vests_to_power(vote_weight, self.blockchain)

            vote_percent = vote_data['percent'] / 100

            efficiency = (((reward - teoric_reward) / teoric_reward) * 100) if vote_weight > 0 else None
            
            # Store basic metrics
            self.data['voting_power'].append(vote_percent)
            self.data['vote_delay'].append(vote_delay)
            self.data['reward'].append(reward)
            self.data['Post'].append(post_data.get('permlink'))
            self.data['Author'].append(post_data.get('author'))
            
            # # Store efficiency metrics
            self.data['like_efficiency'].append(efficiency)
            # self.data['like_efficiency'].append(efficiency)
            
            # Calculate and store author metrics
            author_rewards = float(post_data.get('author_rewards', 0))
            author_reputation = float(post_data.get('author_reputation', 0))
            
            self.data['author_reputation'].append(author_reputation)
            self.data['author_avg_payout'].append(author_rewards)
            
            # Calculate running average efficiency for author
            author_efficiencies = [
                eff for idx, eff in enumerate(self.data['like_efficiency'])
                if self.data['Author'][idx] == post_data.get('author')
            ]
            avg_author_efficiency = sum(author_efficiencies) / len(author_efficiencies) if author_efficiencies else 0
            self.data['author_avg_efficiency'].append(avg_author_efficiency)
            
            # Determine vote success (above average efficiency)
            all_efficiencies = self.data['like_efficiency']
            global_avg_efficiency = sum(all_efficiencies) / len(all_efficiencies) if all_efficiencies else 0
            self.data['success'].append(1 if efficiency > 50 else 0)
            
            logger.debug(f"Processed curation reward for post {post_data.get('permlink')} "
                        f"(Efficiency: {efficiency:.2f}%)")
            
        except Exception as e:
            logger.error(f"Failed to extract metrics: {str(e)}")
            raise

    @retry_on_blockchain_failure()
    def get_author_history(self, author: str, limit: int = 100) -> List[Dict]:
        """Get author's post history for additional analysis."""
        try:
            account = Account(author, blockchain_instance=self.blockchain)
            posts = []
            
            for post in account.history_reverse(limit=limit):
                if post['type'] == 'comment' and not post.get('parent_author'):
                    posts.append(post)
            
            return posts
            
        except Exception as e:
            logger.error(f"Failed to get author history for {author}: {str(e)}")
            raise