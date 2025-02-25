import json
import requests
import logging
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from settings.logging_config import logger
from settings.config import steem_domain, hive_domain
from utils.beem_requests import BlockchainConnector
from command.basic.db import Database

class SocialMediaPublisher:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        self.db_path = config["db_path"]
        self.nodes = config["nodes"]
        self.admin_id = config["admin_id"]
        self.TOKEN = config["TOKEN"]
        self.steem_curator = config["steem_curator"]
        self.steem_curator_posting_key = config["steem_curator_posting_key"]
        self.hive_curator = config["hive_curator"]
        self.hive_curator_posting_key = config["hive_curator_posting_key"]
        self.beem = BlockchainConnector()
        self.last_check_time = defaultdict(lambda: datetime.now(timezone.utc))
        self.published_posts = set()
        self.db = Database()

    def ping_server(self, node_url):
        """Verifica se il nodo è raggiungibile."""
        try:
            response = requests.get(node_url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Error pinging server {node_url}: {e}")
            return False

    def get_posts(self, usernames, node_urls, platform, max_age_minutes=5):
        post_links = []
        current_time = datetime.now(timezone.utc)
        logger.info(f"Recuperando post per {len(usernames)} utenti su {platform}")

        for node_url in node_urls:
            if not self.ping_server(node_url):
                logger.error(f"Impossibile raggiungere il server: {node_url}")
                continue  # Prova il nodo successivo

        headers = {'Content-Type': 'application/json'}

        for username in usernames:
            logger.info(f"Recuperando post per {username} su {platform}")
            data = {
                "jsonrpc": "2.0",
                "method": "condenser_api.get_discussions_by_blog",
                "params": [{"tag": username, "limit": 1}],
                "id": 1
            }
            try:
                response = requests.post(node_url, headers=headers, data=json.dumps(data), timeout=5)
                response.raise_for_status()
                result = response.json().get('result', [])
                for post in result:
                    link = post.get('url')
                    created_time = post.get('created')
                    if link and created_time:
                        post_time = datetime.strptime(created_time, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)
                        post_age = current_time - post_time
                        age_minutes = post_age.total_seconds() / 60
                        last_check_time = self.last_check_time[username]
                        # logger.info(f"Post trovato: {link} - {post_time} - età: {post_age}")
                        # logger.info(f"Ultimo controllo per {username}: {self.last_check_time[username]}")
                        # logger.info(f"Età del post: {post_age.total_seconds() / 60} minuti")
                        # logger.info(f"Massimo tempo di pubblicazione: {max_age_minutes} minuti")

                        if link in self.published_posts:
                            logger.info(f"Il post è già stato pubblicato: {link}")
                            continue

                        if age_minutes <= max_age_minutes:
                            logger.info(f"Post pubblicato di recente: {link}")
                            post_links.append(link)
                            self.published_posts.add(link)
                            self.last_check_time[username] = post_time
                        else:
                            logger.info(f"Post non pubblicato di recente: {link} - età: {post_age}")
            except Exception as e:
                logger.error(f"Errore durante la recuperazione dei post per {username} su {platform}: {e}")

        logger.info(f"Recuperati {len(post_links)} post per {len(usernames)} utenti su {platform}")
        return post_links
    
    def update_user_data(self):
        """Aggiorna i dati degli utenti dal database."""
        steem_usernames = [user['username'] for user in self.db.get_users() if user['platform'] == 'STEEM']
        hive_usernames = [user['username'] for user in self.db.get_users() if user['platform'] == 'HIVE']
        return steem_usernames, hive_usernames

    def publish_posts(self):
        """Ciclo principale per pubblicare i nuovi post trovati."""
        published_links = {"steem": set(), "hive": set()}

        steem_usernames, hive_usernames = self.update_user_data()

        with ThreadPoolExecutor(max_workers=2) as executor:
            while True:
                steem_usernames, hive_usernames = self.update_user_data()

                futures = []
                if steem_usernames:
                    futures.append(
                        executor.submit(self.get_posts, steem_usernames, self.nodes['steem'], 'Steem')
                    )
                if hive_usernames:
                    futures.append(
                        executor.submit(self.get_posts, hive_usernames, self.nodes['hive'], 'Hive')
                    )

                for future, platform in zip(futures, ["steem", "hive"]):
                    links = future.result()
                    new_links = [link for link in links if link not in published_links[platform]]
                    if new_links:
                        domain = steem_domain if platform == "steem" else hive_domain
                        logger.info(f"[{platform.upper()}] New post links: {domain}{new_links}")
                        published_links[platform].update(new_links)
                        for link in new_links:
                            post_link = f"{domain}{link}"
                            if platform.upper() == "STEEM":
                                steem_curator_info = self.beem.get_account_info(self.steem_curator)
                                last_vote_time = steem_curator_info['result'][0]['last_vote_time']
                                steem_voting_power = self.beem.calculate_voting_power(self.steem_curator)
                                telegram_message = f"[{platform.upper()}] (VP: {steem_voting_power})\n{post_link}"
                                self.send_telegram_message(self.TOKEN, self.admin_id, telegram_message)
                                author = self.beem.get_steem_author(post_link)
                                permlink = self.beem.get_steem_permlink(post_link)
                                if steem_voting_power > 89:
                                    time.sleep(30)
                                    self.beem.like_steem_post(voter=self.steem_curator, voted=author, permlink=permlink, private_posting_key=self.steem_curator_posting_key, weight=100)
                                    self.send_telegram_message(self.TOKEN, self.admin_id, "Voted!")
                                else:
                                    self.send_telegram_message(self.TOKEN, self.admin_id, "Not Voted!")
                            elif platform.upper() == "HIVE":
                                hive_curator_info = self.beem.get_hive_profile_info(self.hive_curator)
                                last_vote_time = hive_curator_info['result'][0]['last_vote_time']
                                old_hive_voting_power = hive_curator_info['result'][0]['voting_power'] / 100
                                hive_voting_power = self.beem.calculate_voting_power(last_vote_time, old_hive_voting_power)
                                telegram_message = f"[{platform.upper()}] (VP: {hive_voting_power})\n{post_link}"
                                self.send_telegram_message(self.TOKEN, self.admin_id, telegram_message)
                                author = self.beem.get_hive_author(post_link)
                                permlink = self.beem.get_hive_permlink(post_link)
                                if hive_voting_power > 89:
                                    time.sleep(30)
                                    self.beem.like_hive_post(voter=self.hive_curator, voted=author, permlink=permlink, private_posting_key=self.hive_curator_posting_key, weight=100)
                                    self.send_telegram_message(self.TOKEN, self.admin_id, "Voted!")                                   
                                else:
                                    self.send_telegram_message(self.TOKEN, self.admin_id, "Not Voted!")
                time.sleep(15)  # Controlla ogni 15 secondi

    def send_telegram_message(self, bot_token, chat_id, message):
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={message}"
            response = requests.get(url)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error telegram server {e}")
            return False

if __name__ == '__main__':
    CONFIG_PATH = "config.json"
    publisher = SocialMediaPublisher(CONFIG_PATH)
    publisher.publish_posts()