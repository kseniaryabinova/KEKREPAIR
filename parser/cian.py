#!/usr/bin/env python
# coding: utf-8

# Импорт библиотек
import os
import time
import re
import sqlite3
import pandas as pd

import bs4
from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen
import IPython


# Обработка
def save(filename, content, mode='wb', path_list=['flats']):
    for i in range(1, len(path_list)+1):
        path = os.path.join(*path_list[:i])
        if not os.path.exists(path):
            os.mkdir(path)
    filename = os.path.join(path, filename)
    f = open(filename, mode)
    f.write(content)
    f.close()
    return None


# Настройки
DOMAIN = "https://www.cian.ru/"
newAgentsURL = 'https://www.whatismybrowser.com/guides/the-latest-user-agent/firefox'
newUA = str(BeautifulSoup(requests.get(newAgentsURL).text, 'html.parser').find_all('span', {'class': 'code'})[0])[19:-7]
headers = {'User-Agent': newUA,
           'Content-Type': 'application/x-www-form-urlencoded',
           'Referer': DOMAIN,
           'X-Requested-With': 'XMLHttpRequest'
           }
proxies = {'http' : '109.174.19.134:80', 
           'https': '159.8.114.37:25'}


# Парсинг
def getFromWeb(url, timeout=5, headers=headers, proxies=proxies):
  try:
      page_response = requests.get(url, timeout=timeout, proxies=proxies, headers=headers)
      if page_response.status_code == 200:
          # extract
          return page_response
      else:
          print(page_response.status_code)
          # notify, try again
  except requests.Timeout as e:
      print("It is time to timeout")
      print(str(e))

def download_images(flat_types, flat_pages, time_wait=1):
    for repair_type in flat_types:
        print('repair_type:', repair_type)
        for page_num in flat_pages:
            print('page_num:', page_num)
            url = f"https://www.cian.ru/cat.php?deal_type=rent&engine_version=2&offer_type=flat&p={page_num}&region=1&repair%5B0%5D={repair_type}&room1=1&room2=1&room3=1&type=4"
            r = getFromWeb(url, timeout=5, proxies=None)
            html_data = r.text
            save(f'{page_num}.html', html_data.encode(), mode='wb', path_list=['pages', repair_type])

            posts = []
            articles = BeautifulSoup(html_data, 'html.parser').find_all('article', {'data-name': 'CardComponent'})
            for article in articles:
                flatname = str(article)[str(article).find('href="https://www.cian.ru/rent/flat/')+36:]
                flatname = flatname[:flatname.find('/')]
                # get largest version for 4 photos 
                imgs = [link['src'] for link in  article.find_all('img') if link['src'].endswith('.jpg')]
                imgs = list(set([img[:-5] + '1' + imgs[0][-4:] for img in imgs]))
                # download photos
                for link in imgs:
                    filename = link.split('/')[-1]
                    if not os.path.exists(os.path.join('flats', flatname, filename)):
                        r = getFromWeb(link, timeout=1.5, proxies=None)
                        save(filename, r.content, mode='wb', path_list=['flats', repair_type, flatname])
                        time.sleep(time_wait)
                save('repair_type.txt', repair_type, mode='w', path_list=['flats', repair_type, flatname])
                try:
                    price = str(article.find('span', {'data-mark': 'MainPrice'}))
                    price = price[price.find('>')+1:-14].replace(' ','')
                    save('price.txt', price, mode='w', path_list=['flats', repair_type, flatname])
                    description = (article.text.split('₽')[-1]).split('iEstate')[0]
                    save('description.txt', description, mode='w', path_list=['flats', repair_type, flatname])
                except:
                    pass
                posts.append([imgs, price])
                time.sleep(time_wait)
            time.sleep(time_wait)
        time.sleep(time_wait)
    return None

if __name__ == "__main__": 
	flat_types = ['1','2','3','4']
	flat_pages = ['', '2', '3', '4','5','6','7','8']
	download_images(flat_types, flat_pages, time_wait=1)
