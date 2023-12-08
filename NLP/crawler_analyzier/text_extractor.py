#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 22:11:44 2023

@author: zok
"""

import requests
from time import sleep
import random
import logging
import pandas as pd
import csv

from bs4 import BeautifulSoup

class Scraper:

    def __init__(self):

        self.USER_AGENT_LIST = [
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.7 (KHTML, like Gecko) Chrome/16.0.912.36 Safari/535.7',
            'Mozilla/5.0 (Windows NT 6.2; Win64; x64; rv:16.0) Gecko/16.0 Firefox/16.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/534.55.3 (KHTML, like Gecko) Version/5.1.3 Safari/534.53.10',
            'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1',
            'Mozilla/5.0 (Windows NT 6.3; rv:36.0) Gecko/20100101 Firefox/36.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10; rv:33.0) Gecko/20100101 Firefox/33.0',
            'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20130401 Firefox/31.0'
        ]
        self.filename = None
        self.url = None

    def makeUrl(self):  
        _, url = self.text_extractor()
        return url

    def wait(self, agent, page):
        flag = True
        while True:
            try:
                response = requests.request('GET', self.makeUrl(), timeout=60, headers={'User-Agent': agent})
                logging.info('Got the page!')
                return response
            except requests.ConnectionError:
                if flag:
                    logging.error('Connection lost! Waiting for connection...')
                    flag = False
                sleep(2)
                pass

    def scrap(self):
        i = 1
        agent = self.USER_AGENT_LIST[-1]
        while True:
            logging.info('Requsting new page...')
            
            #50% chance to change agent
            if random.choice(range(100)) <= 50:
                agent = random.choice(self.USER_AGENT_LIST)

            response = self.wait(agent, i)
            if response.status_code != 200:
                if response.status_code == 403:
                    logging.error('Security check not passed :(')
                elif response.status_code == 404:
                    logging.error('Page not found!', self.makeUrl())
                break
            
            if response.status_code == 200:
                result = response.content
    
                soup = BeautifulSoup(result, 'lxml')
                
                # Extract the article title and article text
                title = soup.find_all("h1", class_ =("entry-title", "tdb-title-text"))
                info = soup.find_all("div", class_ =("td-post-content tagdiv-type", "td_block_wrap tdb_single_content tdi_130 td-pb-border-top td_block_template_1 td-post-content tagdiv-type"))
                
                title_list = []
                info_list = []
                for i in range(len(title)):
                    for j in range(len(info)):
                        title_list.append(title[i].get_text())
                        info_list.append(info[j].get_text(strip=True))
                        txt = dict(zip(title_list,info_list))
                
                # Return the article title and text
                return txt
        
            else:
                print('Error: Could not fetch article from URL')
                return None
            
            # if self.out == out:
            #     logging.info('All pages grabbed! Finished!')
            #     break
                  
    def text_extractor(self):
        # Read the input.xlsx file
        df = pd.read_excel('Input.xlsx')
        
        filename = []
        url = []
        # Extract article text for each article URL
        for index, row in df.iterrows():
            url.append(row['URL'])
            # txt = self.scrap()
            # Save the article text in a text file with URL_ID as its file name
            filename.append(str(row['URL_ID']) + '.txt')
            
        return filename, url
        
        self.filename = filename
        self.url = url
            
# df = pd.read_excel('Input.xlsx')
    
# # Extract article text for each article URL
# for index, row in df.iterrows():
#  	url = row['URL']
#  	txt = Scraper().scrap()

#  	# Save the article text in a text file with URL_ID as its file name
#  	filename = str(row['URL_ID']) + '.txt'
#  	with open(filename, 'w') as f:
#  	    writer = csv.writer(f)
#  	    writer.writerow([txt])
