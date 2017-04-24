import urllib2
import re
from bs4 import BeautifulSoup

class fetchPSNProfile:
    def __init__(self, psn_id):
        self.fetch_url = 'https://my.playstation.com/'+str(psn_id)
        self.req = urllib2.Request(self.fetch_url)
        self.response = urllib2.urlopen(self.req)
        self.content = self.response.read()
        self.soup = BeautifulSoup(self.content, "lxml")
        self.re_tag = re.compile(r'<[^>]*>')

    def isExist(self):
        level = self.soup.find(name='div', attrs={'class': 'quantity content level-num'})
        level = self.re_tag.sub('', str(level))
        if level == 'None':
            return False
        else:
            return True

    def getLevel(self):
        level = self.soup.find(name='div', attrs={'class': 'quantity content level-num'})
        level = self.re_tag.sub('', str(level))
        return level

    def getTrophy(self):
        trophy_num = self.soup.find(name='div', attrs={'class': 'quantity content trophy-num'})
        trophy_num = self.re_tag.sub('', str(trophy_num))
        bronze_trophy = self.soup.find(name='div', attrs={'class': 'quantity trophy bronze-trophy'})
        bronze_trophy = self.re_tag.sub('', str(bronze_trophy))
        silver_trophy = self.soup.find(name='div', attrs={'class': 'quantity trophy silver-trophy'})
        silver_trophy = self.re_tag.sub('', str(silver_trophy))
        gold_trophy = self.soup.find(name='div', attrs={'class': 'quantity trophy gold-trophy'})
        gold_trophy = self.re_tag.sub('', str(gold_trophy))
        platinum_trophy = self.soup.find(name='div', attrs={'class': 'quantity trophy platinum-trophy'})
        platinum_trophy = self.re_tag.sub('', str(platinum_trophy))
        trophy = {'trophy_num': trophy_num,
                  'bronze_trophy': bronze_trophy,
                  'silver_trophy': silver_trophy,
                  'gold_trophy': gold_trophy,
                  'platinum_trophy': platinum_trophy}

        return trophy

    def getAvatarUrl(self):
        avatar_tag = self.soup.find(name='img', attrs={'class': 'avatar'})
        avatar_url = 'http:'+avatar_tag['src']

        return avatar_url

    def getLastPlayed(self):
        lastPlayed_tag = self.soup.find(name='h3', attrs={'class': 'lastPlayed'})
        lastPlayed_tag = self.re_tag.sub('', str(lastPlayed_tag))
        lastPlayed = lastPlayed_tag[20:]

        return lastPlayed

